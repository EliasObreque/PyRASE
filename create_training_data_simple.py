# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 15:21:07 2025

@author: mndc5
"""

import pickle
from tqdm import tqdm
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os

from core.sphere_points_test import halton_sphere, rotation_matrix_from_vectors
from core.monitor import show_ray_tracing_fast, plot_sphere_distribution
from core.compute_perturbation_data import compute_ann_data

# Sphere samples
N_SAMPLE = 10000
sphere_vectors = halton_sphere(N_SAMPLE)

# MESH

mesh = pv.Cube(x_length=3, y_length=1, z_length=2)
mesh = mesh.triangulate().clean()
mesh = mesh.subdivide(1, subfilter='linear').clean()

MESH_SCALE = 1 # m, km, mm, cm

mesh.points *= MESH_SCALE
#pl = pv.Plotter()
#_ = pl.add_mesh(mesh, show_edges=True)
#_ = pl.show_grid()
#pl.show()


lx, ly, lz = 3, 1, 2
res_x = res_y = 1000
A_ref = 2

FILENAME = f"./results/data/rect_prism_data_{res_y}_sample_{N_SAMPLE}"

FORCE = True

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

data_mesh = compute_ann_data(mesh, sphere_vectors, sim_data, FILENAME, FORCE,
                             save_analytical=True)

#%%
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

errors = [
    ("CA_error", axes[0], 'blue', "CA Error"),
    ("CN_error", axes[1], 'green', "CN Error"),
    ("CS_error", axes[2], 'red', "CS Error")
]

for error_key, ax, color, title in errors:
    error_data = data_mesh[error_key]
    mean_val = np.mean(error_data)
    std_val = np.std(error_data)
    
    # Histogram
    ax.hist(error_data, bins=30, alpha=0.7, color=color, edgecolor='black')
    
    # Mean line
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
               alpha=0.5, label='Mean')
    
    # Standard deviation lines
    ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2)
    ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2)
    
    # Text box with statistics
    textstr = f'μ = {mean_val:.2e}\nσ = {std_val:.2e}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
            fontsize=16, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(title)
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"results/data/error_rect_prism_model_{res_y}_sample_{N_SAMPLE}.png",  dpi=300, bbox_inches='tight')
plt.show()

#%%

filename = f"results/data/force_drag_rect_prism_model_{res_y}_sample_{N_SAMPLE}.png"


ca_values = np.array(data_mesh["F_drag"])[:, 0] * 1e3
cn_values = np.array(data_mesh["F_drag"])[:, 1]* 1e3
cs_values = np.array(data_mesh["F_drag"])[:, 2]* 1e3

columns_data = [
    (0, ca_values, 'Fx Drag Model', 'x [mN]', "yz"),
    (1, cn_values, 'Fy Drag Model', 'y [mN]', "xz"),
    (2, cs_values, 'Fz Drag Model', 'z [mN]', "xy")
]

plot_sphere_distribution(mesh, data_mesh, columns_data, 
                         filename, show=True)

#%%

filename = f"results/data/force_srp_rect_prism_model_{res_y}_sample_{N_SAMPLE}.png"


ca_values = np.array(data_mesh["F_srp"])[:, 0]* 1e3
cn_values = np.array(data_mesh["F_srp"])[:, 1]* 1e3
cs_values = np.array(data_mesh["F_srp"])[:, 2]* 1e3

columns_data = [
    (0, ca_values, 'Fx SRP Model', 'x [mN]', "yz"),
    (1, cn_values, 'Fy SRP Model', 'y [mN]', "xz"),
    (2, cs_values, 'Fz SRP Model', 'z [mN]', "xy")
]

plot_sphere_distribution(mesh, data_mesh, columns_data, 
                         filename, show=True)


print("Process finished...")
    
    

    
    
    
    
    
    
    
    
    