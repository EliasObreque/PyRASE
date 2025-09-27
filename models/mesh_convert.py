# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 17:33:30 2025

@author: mndc5
"""

import trimesh
import pyvista as pv
from pygltflib import GLTF2


def glb_to_stl(glb_path, stl_path):
    # Load the GLB file
    mesh = trimesh.load(glb_path)
    
    # If the GLB contains multiple meshes, combine them
    if isinstance(mesh, trimesh.Scene):
        # Extract all geometries and combine
        combined = trimesh.util.concatenate([
            geom for geom in mesh.geometry.values()
            if hasattr(geom, 'vertices')
        ])
        mesh = combined
    
    # Export as STL
    mesh.export(stl_path)
    print(f"Converted {glb_path} to {stl_path}")


def glb_to_stl_advanced(glb_path, stl_path):
    # Load GLB with pygltflib for detailed control
    gltf = GLTF2().load(glb_path)
    
    # Load with trimesh for mesh operations
    mesh = trimesh.load(glb_path, force='mesh')
    
    # Handle multiple meshes
    if isinstance(mesh, trimesh.Scene):
        # Get all mesh geometries
        meshes = [geom for geom in mesh.geometry.values() 
                 if isinstance(geom, trimesh.Trimesh)]
        
        if meshes:
            # Combine all meshes
            combined = trimesh.util.concatenate(meshes)
            combined.export(stl_path)
        else:
            print("No valid meshes found in GLB file")
    else:
        mesh.export(stl_path)
    
    print(f"Converted {glb_path} to {stl_path}")
    
    
def visualize_stl_basic(stl_path):
    # Load STL file
    mesh = pv.read(stl_path)
    
    # Create plotter and add mesh
    plotter = pv.Plotter()
    plotter.add_mesh(mesh)
    plotter.show()



model_name = "Aquarius (A)"
#glb_to_stl(model_name + ".glb", model_name + '.stl')

glb_to_stl_advanced(model_name + ".glb", model_name + '.stl')

visualize_stl_basic(model_name + '.stl')
