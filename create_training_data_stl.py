# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 15:21:07 2025

@author: mndc5
"""

import pickle
from tqdm import tqdm
import pyvista as pv
from core.sphere_points_test import halton_sphere

from core.monitor import show_ray_tracing_fast
from core.optimal_ray_tracing import compute_ray_tracing_fast_optimized



# Sphere samples
N_SAMPLE = 20000
sphere_vectors = halton_sphere(N_SAMPLE)

# MESH
reader = pv.get_reader("./models/ALIGN-6U.stl")
mesh = reader.read()
# mesh.rotate_y(90, inplace=True)
mesh = mesh.triangulate().clean()
MESH_SCALE = 1e-3 # m, km, mm, cm

mesh.points *= MESH_SCALE
#pl = pv.Plotter()
#_ = pl.add_mesh(mesh, show_edges=True)
#_ = pl.show_grid()
#pl.show()


# picture
res_x = 1000
res_y = 1000


data_mesh = {}

id_vec = 0 
for r_inout in tqdm(sphere_vectors, "Computation of vector perturbation", N_SAMPLE):
    res_prop = compute_ray_tracing_fast_optimized(mesh, r_inout, res_x, res_y)
    # show_ray_tracing_fast(mesh, res_prop, filename="", show_mesh=True, save_3d=False)
    
    data_mesh[id_vec] = {"r_": r_inout,
                         "mesh_results": res_prop}
    
    
    
with open(f"./results/data/align_data_{res_y}_sample_{N_SAMPLE}", "wb") as file_:
    pickle.dump(data_mesh, file_)


print("Process finished...")
    
    

    
    
    
    
    
    
    
    
    