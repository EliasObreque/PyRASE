"""
Created by Elias Obreque
Date: 23/09/2025
email: els.obrq@gmail.com
"""

import time

import pandas as pd
import pyvista as pv
import numpy as np
from tqdm import tqdm
from scipy.special import erf
import matplotlib.pyplot as plt
from monitor import show_ray_tracing, show_ray_tracing_fast
from ray_tracing import compute_ray_tracing, compute_ray_tracing_fast
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
R_m = 1

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

    A_surf_total = 4 * np.pi * (R_m) ** 2
    A_half_surf = A_surf_total * 0.5
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
        A_surf = pixel_area / cos_th

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
plt.xlabel("Resolution")
plt.ylabel("Absolute error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.yscale("log")
fig.savefig("results/area_errors.png", dpi=300)

fig_time = plt.figure(figsize=(10, 6))
for i, item in enumerate(res_time_calculation):
    plt.plot(resolutions_px, np.array(item)* 1000, '-o', lw=0.7, label=f"Mesh res.: {mesh_res[i]}")
plt.xlabel("Resolution")
plt.ylabel("Calculation Time [ms]")
plt.grid(True)
plt.legend()
plt.tight_layout()
fig_time.savefig("results/time_area_errors.png", dpi=300)
plt.show()

ids_ = np.where(res_resolution_area == np.min(res_resolution_area))
res_ = mesh_res[ids_[0][0]]
res_x = res_y = resolutions_px[ids_[1][0]]
mesh = pv.Sphere(radius=R_m, theta_resolution=res_, phi_resolution=res_, end_theta=360)
mesh = mesh.triangulate().clean()
mesh.rotate_x(-45, inplace=True)
mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
res_prop = compute_ray_tracing_fast(mesh, res_x, res_y)

# === Ray-tracing outputs (already filtered) ===
hits     = res_prop['hit_points']           # (N,3), mm
ray_ids  = res_prop['ray_ids']              # (N,)
cell_ids = res_prop['cell_ids']             # (N,) or None
px_w     = res_prop['pixel_width']          # mm
px_h     = res_prop['pixel_height']         # mm
ray_dir  = res_prop['ray_dirs'][0]          # [0,0,-1]
cos_th   = res_prop['cos_th']
normal_cell = res_prop['cell_normal']

px_area = (px_w * px_h)

# Lever arm for torque (about mesh center)
com_m = np.asarray(mesh.center)
r_m   = hits - com_m                      # (N,3)

# Projected area assignment per-hit.
# Using your approach: A_surf_per_hit = px_area / cos(theta)
A_surf = px_area / cos_th                   # (N,)
# Projected area to the beam (used in SRP closed-form): A_proj = A_surf * cos(theta) = px_area
A_proj = px_area                            # (scalar)

# Build tangent unit vector (orthonormal basis {n, t_hat}) consistent with ray_dir
ref = np.cross(ray_dir, normal_cell)                  # (N,3)
ref_norm = np.linalg.norm(ref, axis=1, keepdims=True)
ref_norm[ref_norm == 0.0] = 1.0
ref = ref / ref_norm
t_hat = np.cross(-ref, normal_cell)
t_hat = t_hat / np.linalg.norm(t_hat, axis=1, keepdims=True)

# === Physical parameters (SI) ===
kB     = 1.3806488e-23
T_wall = 303.15               # K (30 C)
v_mag  = 7800.0               # m/s
rho    = 5e-13                # kg/m^3 (example)
q_vel  = 0.5 * rho * v_mag**2
P_srp  = 4.57e-6              # N/m^2 (solar radiation pressure at 1 AU)
diffuse = 0.2
spec    = 0.8
MM      = 18e-3               # kg/mol
NA      = 6.02214076e23       # 1/mol
m_particle = MM / NA          # kg
vm = np.sqrt(2.0 * kB * T_wall / m_particle)
s_ = v_mag / vm               # speed ratio (scalar)

# === Free-molecular drag model (vectorized) ===
S  = s_ * cos_th
St = s_ * np.sqrt(np.maximum(0.0, 1.0 - cos_th**2))
Pi  = S * np.exp(-(S**2)) + np.sqrt(np.pi) * (S**2 + 0.5) * (1.0 + erf(S))
Chi = np.exp(-(S**2)) + np.sqrt(np.pi) * S * (1.0 + erf(S))

term1 = ((2.0 - diffuse) / np.sqrt(np.pi)) * Pi / (s_**2)
term2 = (diffuse / 2.0) * Chi / (s_**2) * np.sqrt(T_wall / (T_wall + 0.0))
cn = term1 + term2                               # normal coefficient per hit (N)
ct = diffuse * St * Chi / (np.sqrt(np.pi) * s_**2)

# Drag force per hit on the surface patch:
# F_d = q * A_surf * ( cn * n + ct * t_hat )
F_d = q_vel * A_surf[:,None] * (cn[:,None] * normal_cell + ct[:,None] * t_hat)   # (N,3)
T_d = np.cross(r_m, F_d)                                           # (N,3)

# === SRP (Lambert + specular ideal simplified) ===
# Using projected area to the beam (constant per hit in this setup): A_proj = px_area
# Force per hit: F_s = P_srp * A_proj * [ diffuse * d + 2(spec*cos + (diffuse/3)) * n ]
F_s = P_srp * A_proj * (diffuse * ray_dir + (2.0*(spec*cos_th + diffuse/3.0))[:,None] * normal_cell)
T_s = np.cross(r_m, F_s)

# === Sums ===
F_drag_total = F_d.sum(axis=0)
T_drag_total = T_d.sum(axis=0)
F_srp_total  = F_s.sum(axis=0)
T_srp_total  = T_s.sum(axis=0)

print("Drag  F:", F_drag_total, "  Drag  T:", T_drag_total)
print("SRP   F:", F_srp_total,  "  SRP   T:", T_srp_total)

# Optional sanity check vs analytic projected area (sphere of radius R_m)
A_proj_sphere = np.pi * R_m**2
v_vec = ray_dir * v_mag
F_drag_ref = 0.5 * rho * v_mag * v_vec * A_proj_sphere
rel_err = np.linalg.norm(F_drag_total - F_drag_ref)/np.linalg.norm(F_drag_ref)
print("Relative error (drag vs projected-area ref):", rel_err * 100)

if __name__ == '__main__':
    pass
