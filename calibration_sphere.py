"""
Created by Elias Obreque
Date: 23/09/2025
email: els.obrq@gmail.com
"""

import time

import pyvista as pv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.monitor import show_ray_tracing_fast
from src.ray_tracing import compute_ray_tracing_fast
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",                     # generic family
    "font.serif": ["Times New Roman", "Times"], # try exact TNR first
    "font.size": 18,                            # default text size
    "axes.titlesize": 18,                       # axes title
    "axes.labelsize": 18,                       # x/y labels
    "xtick.labelsize": 18,                      # tick labels
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "figure.titlesize": 18,
    # Math text configured to look like Times
    "mathtext.fontset": "stix",                 # STIX resembles Times
    "mathtext.rm": "Times New Roman",
})

# a spherical model of ray tracing to validate that our model and force calculations are accurate

area_paper = 0.2817

R_m = 1
R_m = np.sqrt(area_paper / np.pi)
print(R_m)
A_proj_2d_sphere = np.pi * R_m**2
A_3d_surf_total = 4 * np.pi * (R_m) ** 2
A_half_surf = A_3d_surf_total * 0.5

test_res = False
if test_res:
    mesh_res = [30, 50, 100, 125, 150, 175, 200]
    resolutions_px = [10, 30, 50, 100, 150, 200, 250, 275, 290, 300, 310, 350, 400, 500, 750, 1000]
    
    
    # to remove artifacts on the initialization
    mesh = pv.Sphere(radius=R_m, theta_resolution=10, phi_resolution=10, end_theta=360)
    compute_ray_tracing_fast(mesh, 50, 50)
    # ====================
    
    
    def get_performance_mesh(res_mesh_, show_mesh=False, save_3d=True):
        mesh = pv.Sphere(radius=R_m, theta_resolution=res_mesh_, phi_resolution=res_mesh_, end_theta=360)
        mesh = mesh.triangulate().clean()
        mesh.rotate_x(-45, inplace=True)
        mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
        # mesh.plot(show_edges=True)
    
        ray_direction = np.array([0, 0, -1])
        ray_direction = ray_direction / np.linalg.norm(ray_direction)  # normalise
    

        area_errors = []
        time_calculation = []
        # loop over resolutions to find the optimum
        # optimum is where our total estimated area on pixel array (len(hit_points)*pixel_area) = pi*r^2 with 1% error
        for res in tqdm(resolutions_px, desc="Resolution"):
            res_x = res
            res_y = res
            t0 = time.time()
            res_prop = compute_ray_tracing_fast(mesh, res_x, res_y)
            time_calculation.append(time.time() - t0)
            print("time 2:", time.time() - t0)
            if show_mesh or save_3d:
                filename = f"./results/mesh_res_{res_mesh_}_and_px_res_{res}.png"
                show_ray_tracing_fast(mesh, res_prop, filename=filename, show_mesh=show_mesh, save_3d=save_3d)
    
            hit_points = res_prop['hit_points']
            pixel_width = res_prop['pixel_width']
            pixel_height = res_prop['pixel_height']
            pixel_area = pixel_width * pixel_height
            cos_th = res_prop['cos_th']
            A_surf = -pixel_area / cos_th
    
            area = np.sum(A_surf)
            area_error = np.abs((area - A_half_surf) / A_half_surf) * 100
    
            area_errors.append(area_error)
            print("\n Absolute error:", area_error, "[%]", area, A_half_surf, "\n")
        return area_errors, time_calculation
    
    
    res_resolution_area = []
    res_time_calculation = []
    for res_i in mesh_res:
        area_errors_i, time_calculation_i = get_performance_mesh(res_i, show_mesh=False, save_3d=True)
        res_resolution_area.append(area_errors_i)
        res_time_calculation.append(time_calculation_i)
    
    
    fig = plt.figure(figsize=(10, 6))
    for i, item in enumerate(res_resolution_area):
        plt.plot(resolutions_px, item, '-o', lw=0.7, label=f"Mesh res.: {mesh_res[i]}")
    plt.xlabel("Pixels Resolution")
    plt.ylabel("Absolute Area error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.yscale("log")
    fig.savefig("results/area_errors.png", dpi=300)
    
    fig_time = plt.figure(figsize=(10, 6))
    for i, item in enumerate(res_time_calculation):
        plt.plot(resolutions_px, np.array(item)* 1000, '-o', lw=0.7, label=f"Mesh res.: {mesh_res[i]}")
    plt.xlabel("Pixels Resolution")
    plt.ylabel("Calculation Time [ms]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig_time.savefig("results/time_area_errors.png", dpi=300)
    plt.show()
    
    ids_ = np.where(res_resolution_area == np.min(res_resolution_area))
    res_ = mesh_res[ids_[0][0]]
    res_x = res_y = resolutions_px[ids_[1][0]]
    
    



if __name__ == '__main__':
    pass
