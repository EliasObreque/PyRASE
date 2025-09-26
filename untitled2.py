# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 16:40:53 2025

@author: mndc5
"""

import sys
import json
import os
from typing import Dict, List, Any
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QLabel, QSplitter, QTextEdit,
    QToolBar, QStatusBar, QFileDialog, QMessageBox
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl, Slot, QObject, Signal
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtGui import QAction

class WorkflowBridge(QObject):
    """Simple bridge between Python and JavaScript"""
    
    workflow_changed = Signal(str)
    node_executed = Signal(str, str)
    
    def __init__(self):
        super().__init__()
        self.current_workflow = {}
        
    @Slot(str)
    def save_workflow(self, workflow_data):
        """Save workflow data"""
        try:
            self.current_workflow = json.loads(workflow_data)
            node_count = len(self.current_workflow.get('nodes', []))
            print(f"Workflow saved: {node_count} nodes")
            self.workflow_changed.emit(f"Saved {node_count} nodes")
        except Exception as e:
            print(f"Save error: {e}")
    
    @Slot(str, str, result=str)
    def execute_node(self, node_id, input_data):
        """Execute a node"""
        try:
            print(f"Executing: {node_id}")
            result = {"node_id": node_id, "success": True, "output": f"Processed {node_id}"}
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e), "success": False})
    
    @Slot(str)
    def log_message(self, message):
        """Log from frontend"""
        print(f"JS: {message}")

class SimpleWorkflowEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Workflow Editor")
        self.setGeometry(100, 100, 1400, 900)
        
        # Setup bridge
        self.bridge = WorkflowBridge()
        self.web_channel = QWebChannel()
        self.web_channel.registerObject("pyBridge", self.bridge)
        
        self._setup_ui()
        self._load_editor()
        
        # Connect signals
        self.bridge.workflow_changed.connect(self._on_workflow_changed)
        
    def _setup_ui(self):
        # Toolbar
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        actions = [
            ("New", self._new_workflow),
            ("Save", self._save_workflow),
            ("Clear", self._clear_workflow),
            ("Test", self._test_create)
        ]
        
        for name, func in actions:
            action = QAction(name, self)
            action.triggered.connect(func)
            toolbar.addAction(action)
        
        # Main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Web view
        self.web_view = QWebEngineView()
        self.web_view.page().setWebChannel(self.web_channel)
        layout.addWidget(self.web_view)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: #ffffff; }
            QToolBar { background-color: #3c3c3c; border: none; padding: 5px; }
            QToolBar QToolButton { 
                background-color: #4a4a4a; color: #ffffff; padding: 8px 15px; 
                border-radius: 3px; margin: 2px; font-weight: bold;
            }
            QToolBar QToolButton:hover { background-color: #0078d4; }
            QStatusBar { background-color: #3c3c3c; color: #ffffff; }
        """)
    
    def _load_editor(self):
        """Load the simple editor HTML"""
        html_content = self._generate_html()
        self.web_view.setHtml(html_content)
    
    def _generate_html(self):
        """Generate minimal HTML editor"""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Simple Workflow Editor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: #1e1e1e; 
            color: #ffffff; 
            overflow: hidden; 
            height: 100vh; 
        }
        
        .container { display: flex; height: 100vh; }
        
        .sidebar { 
            width: 300px; 
            background: #2d2d2d; 
            border-right: 1px solid #404040; 
            padding: 20px; 
            overflow-y: auto; 
        }
        
        .sidebar h3 { 
            color: #ffffff; 
            margin: 20px 0 10px 0; 
            font-size: 14px; 
            text-transform: uppercase; 
            letter-spacing: 1px; 
        }
        
        .node-item { 
            background: #3c3c3c; 
            border: 1px solid #555; 
            border-radius: 6px; 
            padding: 15px; 
            margin: 8px 0; 
            cursor: pointer; 
            transition: all 0.2s; 
        }
        
        .node-item:hover { 
            background: #4a4a4a; 
            border-color: #0078d4; 
            transform: translateY(-2px); 
        }
        
        .node-item:active { 
            background: #0078d4; 
            transform: scale(0.98); 
        }
        
        .node-title { 
            font-weight: bold; 
            margin-bottom: 5px; 
        }
        
        .node-desc { 
            font-size: 12px; 
            color: #cccccc; 
        }
        
        .canvas { 
            flex: 1; 
            background: #1e1e1e; 
            position: relative; 
            overflow: hidden; 
        }
        
        .workflow-node { 
            position: absolute; 
            background: #2d2d2d; 
            border: 2px solid #555; 
            border-radius: 8px; 
            min-width: 180px; 
            cursor: move; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.3); 
        }
        
        .workflow-node:hover { 
            border-color: #0078d4; 
            box-shadow: 0 6px 20px rgba(0,120,212,0.3); 
        }
        
        .workflow-node.dragging { 
            transform: scale(1.05); 
            z-index: 1000; 
        }
        
        .node-header { 
            background: linear-gradient(135deg, #0078d4, #106ebe); 
            color: white; 
            padding: 12px 15px; 
            font-weight: bold; 
            font-size: 13px; 
            border-radius: 6px 6px 0 0; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
        }
        
        .node-body { 
            padding: 15px; 
            font-size: 12px; 
            color: #e0e0e0; 
        }
        
        .delete-btn { 
            background: none; 
            border: none; 
            color: #ff6b6b; 
            cursor: pointer; 
            font-size: 16px; 
            padding: 2px 5px; 
            border-radius: 3px; 
        }
        
        .delete-btn:hover { 
            background: rgba(255,107,107,0.2); 
        }
        
        .status { 
            position: absolute; 
            top: 20px; 
            right: 20px; 
            background: rgba(45,45,45,0.9); 
            padding: 10px 15px; 
            border-radius: 6px; 
            font-size: 12px; 
            color: #cccccc; 
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h3>Available Nodes</h3>
            
            <div class="node-item" data-type="input">
                <div class="node-title">Input Node</div>
                <div class="node-desc">Receive data input</div>
            </div>
            
            <div class="node-item" data-type="process">
                <div class="node-title">Process Node</div>
                <div class="node-desc">Process and transform data</div>
            </div>
            
            <div class="node-item" data-type="output">
                <div class="node-title">Output Node</div>
                <div class="node-desc">Send data output</div>
            </div>
            
            <div class="node-item" data-type="condition">
                <div class="node-title">Condition Node</div>
                <div class="node-desc">Conditional logic</div>
            </div>
            
            <div class="node-item" data-type="transform">
                <div class="node-title">Transform Node</div>
                <div class="node-desc">Data transformation</div>
            </div>
            
            <div class="node-item" data-type="filter">
                <div class="node-title">Filter Node</div>
                <div class="node-desc">Filter and validate data</div>
            </div>
        </div>
        
        <div class="canvas" id="canvas">
            <div class="status" id="status">
                Nodes: 0 | Ready
            </div>
        </div>
    </div>

    <script>
        console.log('Simple Workflow Editor Loading...');
        
        // Simple global state
        var nodes = [];
        var nodeCounter = 0;
        var isDragging = false;
        
        // Python bridge
        var pyBridge = null;
        
        if (typeof qt !== 'undefined' && qt.webChannelTransport) {
            new QWebChannel(qt.webChannelTransport, function (channel) {
                pyBridge = channel.objects.pyBridge;
                console.log('Connected to Python');
                initialize();
            });
        } else {
            console.log('No Python bridge - using fallback');
            pyBridge = {
                save_workflow: function(data) { console.log('Save:', data.length); },
                execute_node: function(id, data) { return JSON.stringify({success: true}); },
                log_message: function(msg) { console.log('Log:', msg); }
            };
            initialize();
        }
        
        function log(msg) {
            console.log('WORKFLOW: ' + msg);
            if (pyBridge && pyBridge.log_message) {
                pyBridge.log_message(msg);
            }
        }
        
        function updateStatus() {
            document.getElementById('status').textContent = 'Nodes: ' + nodes.length + ' | Ready';
        }
        
        function initialize() {
            log('Initializing editor');
            setupEventListeners();
            updateStatus();
            
            // Create test node
            setTimeout(function() {
                createNode('input', 300, 150);
                log('Created initial test node');
            }, 500);
        }
        
        function setupEventListeners() {
            log('Setting up events');
            
            // Sidebar clicks
            var nodeItems = document.querySelectorAll('.node-item');
            for (var i = 0; i < nodeItems.length; i++) {
                (function(item) {
                    item.addEventListener('click', function(e) {
                        e.preventDefault();
                        var type = item.getAttribute('data-type');
                        log('Creating node: ' + type);
                        
                        // Flash effect
                        item.style.backgroundColor = '#00aa00';
                        setTimeout(function() {
                            item.style.backgroundColor = '';
                        }, 150);
                        
                        // Create node at center
                        createNode(type, 400 + Math.random() * 200, 200 + Math.random() * 200);
                    });
                })(nodeItems[i]);
            }
            
            log('Event listeners ready');
        }
        
        function createNode(type, x, y) {
            var nodeId = 'node_' + (++nodeCounter);
            log('Creating node: ' + nodeId + ' (' + type + ')');
            
            var node = {
                id: nodeId,
                type: type,
                x: x || 300,
                y: y || 150
            };
            
            nodes.push(node);
            renderNode(node);
            saveWorkflow();
            updateStatus();
        }
        
        function renderNode(node) {
            var nodeEl = document.createElement('div');
            nodeEl.className = 'workflow-node';
            nodeEl.style.left = node.x + 'px';
            nodeEl.style.top = node.y + 'px';
            nodeEl.setAttribute('data-id', node.id);
            
            var title = getNodeTitle(node.type);
            
            nodeEl.innerHTML = 
                '<div class="node-header">' +
                    '<span>' + title + '</span>' +
                    '<button class="delete-btn" onclick="deleteNode(\'' + node.id + '\')">×</button>' +
                '</div>' +
                '<div class="node-body">' +
                    'Type: ' + node.type + '<br>' +
                    'ID: ' + node.id +
                '</div>';
            
            // Make draggable
            makeDraggable(nodeEl, node);
            
            // Add flash effect
            nodeEl.style.border = '3px solid #00ff00';
            setTimeout(function() {
                nodeEl.style.border = '2px solid #555';
            }, 800);
            
            document.getElementById('canvas').appendChild(nodeEl);
            log('Rendered node: ' + node.id);
        }
        
        function makeDraggable(element, node) {
            var startX, startY, startNodeX, startNodeY;
            
            element.addEventListener('mousedown', function(e) {
                if (e.target.classList.contains('delete-btn')) return;
                
                isDragging = true;
                element.classList.add('dragging');
                
                startX = e.clientX;
                startY = e.clientY;
                startNodeX = node.x;
                startNodeY = node.y;
                
                log('Started dragging: ' + node.id);
                
                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
                
                e.preventDefault();
            });
            
            function onMouseMove(e) {
                if (!isDragging) return;
                
                var deltaX = e.clientX - startX;
                var deltaY = e.clientY - startY;
                
                node.x = startNodeX + deltaX;
                node.y = startNodeY + deltaY;
                
                // Keep in bounds
                node.x = Math.max(10, Math.min(node.x, window.innerWidth - 200));
                node.y = Math.max(10, Math.min(node.y, window.innerHeight - 100));
                
                element.style.left = node.x + 'px';
                element.style.top = node.y + 'px';
            }
            
            function onMouseUp() {
                if (!isDragging) return;
                
                isDragging = false;
                element.classList.remove('dragging');
                
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
                
                saveWorkflow();
                log('Finished dragging: ' + node.id + ' to (' + Math.round(node.x) + ', ' + Math.round(node.y) + ')');
            }
        }
        
        function getNodeTitle(type) {
            var titles = {
                'input': 'Input Node',
                'process': 'Process Node',
                'output': 'Output Node',
                'condition': 'Condition Node',
                'transform': 'Transform Node',
                'filter': 'Filter Node'
            };
            return titles[type] || type;
        }
        
        function deleteNode(nodeId) {
            log('Deleting node: ' + nodeId);
            
            // Remove from array
            nodes = nodes.filter(function(n) { return n.id !== nodeId; });
            
            // Remove from DOM
            var element = document.querySelector('[data-id="' + nodeId + '"]');
            if (element) {
                element.remove();
            }
            
            saveWorkflow();
            updateStatus();
        }
        
        function saveWorkflow() {
            var workflow = { nodes: nodes };
            if (pyBridge && pyBridge.save_workflow) {
                pyBridge.save_workflow(JSON.stringify(workflow));
            }
        }
        
        function clearAll() {
            log('Clearing all nodes');
            nodes = [];
            var canvas = document.getElementById('canvas');
            var nodeElements = canvas.querySelectorAll('.workflow-node');
            for (var i = 0; i < nodeElements.length; i++) {
                nodeElements[i].remove();
            }
            saveWorkflow();
            updateStatus();
        }
        
        function createTestNode() {
            createNode('process', 500, 300);
        }
        
        // Global functions for toolbar
        window.clearAll = clearAll;
        window.createTestNode = createTestNode;
        
        log('Simple Workflow Editor Ready');
    </script>
</body>
</html>
        """
    
    def _new_workflow(self):
        """Create new workflow"""
        self.web_view.page().runJavaScript("clearAll();")
        self.status_bar.showMessage("New workflow created")
    
    def _save_workflow(self):
        """Save workflow"""
        self.status_bar.showMessage("Workflow saved")
    
    def _clear_workflow(self):
        """Clear workflow"""
        self.web_view.page().runJavaScript("clearAll();")
        self.status_bar.showMessage("Workflow cleared")
    
    def _test_create(self):
        """Test create node"""
        self.web_view.page().runJavaScript("createTestNode();")
        self.status_bar.showMessage("Test node created")
    
    def _on_workflow_changed(self, message):
        """Handle workflow change"""
        self.status_bar.showMessage(message)

def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        editor = SimpleWorkflowEditor()
        editor.show()
        
        if QApplication.instance() == app and app is not None:
            return app.exec()
        else:
            return editor
            
    except ImportError:
        print("QWebEngineWidgets not available")
        return None

if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        sys.exit(result)