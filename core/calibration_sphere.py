# -*- coding: utf-8 -*-
"""
Created by Elias Obreque
Date: 23/09/2025
Modified: Statistical analysis with standard deviation
email: els.obrq@gmail.com
"""

import time
import os
import pyvista as pv
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from core.monitor import show_ray_tracing_fast
from core.optimal_ray_tracing import compute_ray_tracing_fast_optimized
import matplotlib as mpl
from rich import print

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "figure.titlesize": 18,
    "mathtext.fontset": "stix",
    "mathtext.rm": "Times New Roman",
})

# Pastel color palette
PASTEL_COLORS = [
    '#6B8CD4',  # Blue
    '#7CAC9D',  # Teal
    '#A4B86E',  # Yellow-green
    '#D8944D',  # Orange
    '#D87F7F',  # Pink/Red
    '#9B7CB4',  # Purple
    '#D4A86B',  # Tan
]

# Reference area from paper
area_paper = 0.2817

R_m = np.sqrt(area_paper / np.pi)
print(f"Sphere radius: {R_m}")
A_proj_2d_sphere = np.pi * R_m**2
A_3d_surf_total = 4 * np.pi * (R_m) ** 2
A_half_surf = A_3d_surf_total * 0.5

# Number of random samples for statistical analysis
N_SAMPLES = 10

test_res = True
file_data = "results/calibration_statistical.pkl"


def generate_random_unit_vectors(n):
    """Generate n random unit vectors uniformly distributed on a sphere."""
    vectors = []
    for _ in range(n):
        # Use spherical coordinates for uniform distribution
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        vec = np.array([x, y, z])
        vectors.append(vec / np.linalg.norm(vec))
    return vectors


if test_res or not os.path.exists(file_data):
    mesh_res = [50, 100, 250, 500]
    resolutions_px = [100, 200, 300, 400, 500, 750, 1000]

    # Warm the code to remove artifacts on the initialization
    mesh = pv.Sphere(radius=R_m, theta_resolution=10, phi_resolution=10, end_theta=360)
    compute_ray_tracing_fast_optimized(mesh, np.array([0, 0, -1]), 50, 50, 50)
    # ====================

    def get_performance_mesh_statistical(res_mesh_, n_samples=N_SAMPLES, show_mesh=False, save_3d=False):
        """
        Calculate area errors and timing for multiple random ray directions.
        Returns mean and std for both metrics.
        """
        mesh = pv.Sphere(radius=R_m, theta_resolution=res_mesh_, phi_resolution=res_mesh_, end_theta=360)
        mesh = mesh.triangulate().clean()
        mesh.rotate_x(-45, inplace=True)
        mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)

        # Generate random unit vectors for this mesh resolution
        ray_directions = generate_random_unit_vectors(n_samples)

        area_errors_mean = []
        area_errors_std = []
        time_mean = []
        time_std = []

        for res in tqdm(resolutions_px, desc=f"Mesh res {res_mesh_}"):
            res_x = res
            res_y = res
            
            # Collect results for all ray directions
            area_errors_samples = []
            time_samples = []
            
            for ray_dir in ray_directions:
                t0 = time.time()
                res_prop = compute_ray_tracing_fast_optimized(mesh, ray_dir, res_x, res_y)
                elapsed = time.time() - t0
                time_samples.append(elapsed)
                
                # Calculate area error
                pixel_width = res_prop['pixel_width']
                pixel_height = res_prop['pixel_height']
                pixel_area = pixel_width * pixel_height
                cos_th = res_prop['cos_th']
                A_surf = -pixel_area / cos_th
                
                area = np.sum(A_surf)
                area_error = np.abs((area - A_half_surf) / A_half_surf) * 100
                area_errors_samples.append(area_error)
            
            # Calculate statistics
            area_errors_mean.append(np.mean(area_errors_samples))
            area_errors_std.append(np.std(area_errors_samples))
            time_mean.append(np.mean(time_samples))
            time_std.append(np.std(time_samples))
            
            print(f"[yellow]Res {res}: Area error = {np.mean(area_errors_samples):.3f} +/- {np.std(area_errors_samples):.3f} %[/yellow]")

        return {
            'area_mean': area_errors_mean,
            'area_std': area_errors_std,
            'time_mean': time_mean,
            'time_std': time_std
        }

    results = {}
    for res_i in mesh_res:
        print(f"\n[bold cyan]Processing mesh resolution: {res_i}[/bold cyan]")
        results[res_i] = get_performance_mesh_statistical(res_i, n_samples=N_SAMPLES, show_mesh=False, save_3d=False)

    data = {
        'results': results,
        'mesh_res': mesh_res,
        'res_px': resolutions_px,
        'n_samples': N_SAMPLES
    }

    os.makedirs("results", exist_ok=True)
    with open(file_data, "wb") as f:
        pickle.dump(data, f)
else:
    with open(file_data, "rb") as f:
        data = pickle.load(f)

results = data.get("results", {})
mesh_res = data.get("mesh_res", [])
resolutions_px = data.get("res_px", [])
n_samples = data.get("n_samples", N_SAMPLES)

# Plot Area Errors with Standard Deviation
fig, ax = plt.subplots(figsize=(8, 4))
for i, res_m in enumerate(mesh_res):
    color = PASTEL_COLORS[i % len(PASTEL_COLORS)]
    res_data = results[res_m]
    mean = np.array(res_data['area_mean'])
    std = np.array(res_data['area_std'])
    
    ax.plot(resolutions_px, mean, '-o', lw=1.5, color=color, label=f"Mesh res.: {res_m}", markersize=5)
    ax.fill_between(resolutions_px, mean - std, mean + std, alpha=0.25, color=color)

ax.set_xlabel("Pixels Resolution")
ax.set_ylabel("Absolute Area Error [%]")
ax.grid(True, alpha=0.3)
ax.legend(loc='best')
fig.tight_layout()
fig.savefig("results/area_errors_statistical.png", dpi=300)

# Plot Calculation Time with Standard Deviation
fig_time, ax_time = plt.subplots(figsize=(8, 4))
for i, res_m in enumerate(mesh_res):
    color = PASTEL_COLORS[i % len(PASTEL_COLORS)]
    res_data = results[res_m]
    mean = np.array(res_data['time_mean']) * 1000  # Convert to ms
    std = np.array(res_data['time_std']) * 1000
    
    ax_time.plot(resolutions_px, mean, '-o', lw=1.5, color=color, label=f"Mesh res.: {res_m}", markersize=5)
    ax_time.fill_between(resolutions_px, mean - std, mean + std, alpha=0.25, color=color)

ax_time.set_xlabel("Pixels Resolution")
ax_time.set_ylabel("Calculation Time [ms]")
ax_time.grid(True, alpha=0.3)
ax_time.legend(loc='best')
fig_time.tight_layout()
fig_time.savefig("results/time_area_errors_statistical.png", dpi=300)

plt.show()

if __name__ == '__main__':
    pass