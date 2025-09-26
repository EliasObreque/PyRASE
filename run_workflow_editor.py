import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import QUrl, QObject, Signal, Slot, Qt
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# -------- Bridge Python <-> JS ---------
class WorkflowBridge(QObject):
    workflow_changed = Signal(str)        # message
    node_executed = Signal(str, str)      # node_id, result_json
    open_config = Signal(str, str)        # node_id, node_type

    def __init__(self):
        super().__init__()
        self.current_workflow = {"nodes": [], "connections": []}
        self.execution_results = {}

    # called by JS: pyBridge.save_workflow(JSON)
    @Slot(str)
    def save_workflow(self, workflow_json: str) -> None:
        try:
            self.current_workflow = json.loads(workflow_json) if workflow_json else {}
            msg = f"Saved: {len(self.current_workflow.get('nodes', []))} nodes, {len(self.current_workflow.get('connections', []))} links"
            self.workflow_changed.emit(msg)
        except Exception as e:
            self.workflow_changed.emit(f"Save error: {e}")

    # called by JS: pyBridge.execute_node(id, json)
    @Slot(str, str, result=str)
    def execute_node(self, node_id: str, input_data: str) -> str:
        # stub execution
        try:
            payload = json.loads(input_data) if input_data else {}
        except Exception:
            payload = {"raw": input_data}
        result = {"success": True, "echo": payload}
        self.execution_results[node_id] = result
        self.node_executed.emit(node_id, json.dumps(result))
        return json.dumps(result)

    # called by JS on node double‑click: pyBridge.open_config_dialog(node_id, node_type)
    @Slot(str, str)
    def open_config_dialog(self, node_id: str, node_type: str) -> None:
        self.open_config.emit(node_id, node_type)


# -------- Simple config dialogs ---------
class FileReaderDialog(QDialog):
    def __init__(self, node_id: str, current_path: str | None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Configure File Reader — {node_id}")
        self.setModal(True)
        self.setMinimumWidth(480)

        self.path_edit = QLineEdit(current_path or "")
        browse_btn = QPushButton("Browse…")
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")

        browse_btn.clicked.connect(self._browse)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        row = QHBoxLayout()
        row.addWidget(self.path_edit)
        row.addWidget(browse_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("File path:"))
        lay.addLayout(row)

        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        lay.addLayout(btns)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select file")
        if path:
            self.path_edit.setText(path)

    @property
    def file_path(self) -> str:
        return self.path_edit.text().strip()


# -------- Main window ---------
class WorkflowEditorWindow(QMainWindow):
    def __init__(self, html_path: Path | None = None):
        super().__init__()
        self.setWindowTitle("Workflow Editor — PySide6 host")
        self.resize(1400, 900)

        # web view + channel
        self.view = QWebEngineView(self)
        self.setCentralWidget(self.view)

        self.bridge = WorkflowBridge()
        self.channel = QWebChannel(self.view.page())
        self.channel.registerObject("pyBridge", self.bridge)
        self.view.page().setWebChannel(self.channel)

        # toolbar
        tb = QToolBar("Main", self)
        self.addToolBar(Qt.TopToolBarArea, tb)
        tb.addAction("New", self._new)
        tb.addAction("Save", self._save)
        tb.addAction("Load", self._load)
        tb.addSeparator()
        tb.addAction("Run", self._run)
        tb.addAction("Clear", self._clear)
        tb.addAction("Example", self._example)
        tb.addSeparator()
        tb.addAction("Export JSON", self._export_json)

        # status bar
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)

        # signals
        self.bridge.workflow_changed.connect(self.status.showMessage)
        self.bridge.node_executed.connect(self._on_node_executed)
        self.bridge.open_config.connect(self._on_open_config)

        # load html
        self._load_html(html_path)

        # Inject helper used to update node config from Python
        self._inject_update_helper()

    # ---- loading ----
    def _load_html(self, html_path: Path | None):
        if html_path and html_path.exists():
            self.view.setUrl(QUrl.fromLocalFile(str(html_path)))
        else:
            # Fall back to canvas filename candidates in CWD
            candidates = [
                Path.cwd() / "workflow_editor_enhanced.html",
                Path.cwd() / "workflow_editor_fixed.html",
                Path.cwd() / "Workflow Editor Enhanced.html",
                Path.cwd() / "Workflow Editor Fixed.html",
            ]
            for p in candidates:
                if p.exists():
                    self.view.setUrl(QUrl.fromLocalFile(str(p)))
                    break
            else:
                # As last resort, instruct user
                html_msg = (
                    "No local HTML found. Save your HTML as 'workflow_editor_enhanced.html' or pass a path."
                )
                self.status.showMessage(html_msg)

    def _inject_update_helper(self):
        # Exposes a function inside the page to update a node config from Python
        js = r"""
        (function(){
          window.__updateNodeConfigFromHost = function(nodeId, newConfig){
            try{
              if(typeof nodes === 'undefined') return false;
              const n = nodes.find(n=>n.id===nodeId);
              if(!n) return false;
              n.config = Object.assign({}, n.config || {}, newConfig || {});
              if(typeof saveWorkflow === 'function') saveWorkflow();
              return true;
            }catch(e){ return false; }
          }
        })();
        """
        self.view.page().runJavaScript(js)

    # ---- toolbar actions ----
    def _new(self):
        self._eval_js("clearWorkflow();")

    def _save(self):
        self._eval_js("saveWorkflow();")

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load workflow JSON", filter="JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                wf = json.load(f)
            payload = json.dumps(wf)
            # Replace in-page graph
            self._eval_js(f"nodes = {json.dumps(wf.get('nodes', []))}; connections = {json.dumps(wf.get('connections', []))}; clearWorkflow();")
            # Re-render nodes by asking page to rebuild example and then overwrite
            self._eval_js(
                "(function(){ const c=document.getElementById('canvas'); if(c){ c.querySelectorAll('.workflow-node').forEach(n=>n.remove()); }})();"
            )
            # Ask page to render nodes again by calling its create function for each
            for n in wf.get("nodes", []):
                # Create and then set position + config
                t = n.get("type", "http-request")
                nid = n.get("id", "")
                self._eval_js(f"(function(){{ var nn = createNode('{t}', {n.get('x', 100)}, {n.get('y', 100)}); }})();")
                if n.get("config"):
                    self._eval_js(f"window.__updateNodeConfigFromHost('{nid}', {json.dumps(n['config'])});")
            # Recreate connections
            for c in wf.get("connections", []):
                js_conn = (
                    "(function(){ if(typeof createConnection==='function'){"
                    f" createConnection({{'nodeId':'{c['from']['nodeId']}', 'portName':'{c['from']['port']}', 'isOutput':true}},"
                    f"                 {{'nodeId':'{c['to']['nodeId']}',   'portName':'{c['to']['port']}',   'isOutput':false}}); }} )();"
                )
                self._eval_js(js_conn)
            self.status.showMessage(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "Load error", str(e))

    def _run(self):
        self._eval_js("runWorkflow();")

    def _clear(self):
        self._eval_js("clearWorkflow();")

    def _example(self):
        self._eval_js("createExample();")

    def _export_json(self):
        if not self.bridge.current_workflow:
            QMessageBox.information(self, "Export", "Nothing to export. Use Save first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export workflow", "workflow.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.bridge.current_workflow, f, indent=2)
            self.status.showMessage(f"Exported: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "Export error", str(e))

    # ---- bridge callbacks ----
    def _on_node_executed(self, node_id: str, result_json: str) -> None:
        ok = json.loads(result_json).get("success", False)
        self.status.showMessage(f"Node {node_id} executed: {'OK' if ok else 'ERR'}")

    def _on_open_config(self, node_id: str, node_type: str) -> None:
        # Fetch current config from bridge cache (optional)
        cfg = {}
        for n in self.bridge.current_workflow.get("nodes", []):
            if n.get("id") == node_id:
                cfg = n.get("config", {})
                break

        if node_type == "file-reader":
            dlg = FileReaderDialog(node_id, cfg.get("file_path"))
            if dlg.exec() == QDialog.Accepted:
                new_path = dlg.file_path
                # Update JS node config and re-save
                self._eval_js(f"window.__updateNodeConfigFromHost('{node_id}', {{file_path: {json.dumps(new_path)} }});")
        else:
            # Generic info dialog
            QMessageBox.information(self, "Config", f"No custom UI for type: {node_type}")

    # ---- helpers ----
    def _eval_js(self, js: str) -> None:
        self.view.page().runJavaScript(js)


# -------- entry point ---------
def main():
    app = QApplication.instance() or QApplication(sys.argv)

    html_arg = None
    if len(sys.argv) > 1:
        html_arg = Path(sys.argv[1])
    win = WorkflowEditorWindow(html_arg)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
