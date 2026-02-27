# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 15:21:07 2025

@author: mndc5
"""

import pickle
from tqdm import tqdm
import pyvista as pv
from core.sphere_points_test import halton_sphere
from core.compute_perturbation_data import compute_ann_data
from core.monitor import show_ray_tracing_fast
from core.optimal_ray_tracing import compute_ray_tracing_fast_optimized


# MESH
reader = pv.get_reader("./models/Aqua+(B).stl")
mesh = reader.read()
# mesh.rotate_y(90, inplace=True)
MESH_SCALE = 1e-0 # m, km, mm, cm
lx, ly, lz = mesh.bounds_size
mesh.points *= MESH_SCALE
pl = pv.Plotter()
_ = pl.add_mesh(mesh, show_edges=True)
_ = pl.show_grid()
pl.show()

# picture
res_x = 1000
res_y = 1000
# Sphere samples
N_SAMPLE = 20000
sphere_vectors = halton_sphere(N_SAMPLE)

OUT_FILENAME = f"./results/data/aqua_b_data_{res_y}_sample_{N_SAMPLE}"


A_ref = (lx*ly + ly*lz + lz*lz) / 3
data_mesh = {}

sim_data = {
    "spec_srp": 0.2,           # Mostly diffuse reflection (typical for spacecraft)
    "diffuse_srp": 0.6,        # 60% diffuse
    "v_inf": 7800.0,       # Orbital velocity at ~400 km altitude [m/s]
    "alt_km": 400.0,       # ISS altitude [km]
    "time_str": "2025-10-23T12:00:00",
    "sigma_N": 0.8,       # Nearly complete accommodation
    "sigma_T": 0.5,       # High tangential accommodation
    "T_wall": 300.0,       # Average spacecraft surface temp [K]
    "A_ref": A_ref,          # Reference area (e.g., cross-sectional) [m²]
    "lx": lx,
    "ly": ly,
    "lz": lz
}


FORCE = True

data_mesh = compute_ann_data(mesh, sphere_vectors, sim_data, OUT_FILENAME, FORCE,
                             save_analytical=False)
    

print("Process finished...")
    
    

    
    
    
    
    
    
    
    
    