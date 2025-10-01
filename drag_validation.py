"""
Created by Elias Obreque
Date: 01/10/2025
email: els.obrq@gmail.com
"""
import numpy as np
from scipy.special import erf
import pyvista as pv
from src.monitor import show_ray_tracing_fast
from src.ray_tracing import compute_ray_tracing_fast
import time

res_ = 50
res_x = res_y = 100
A_proj_2d_sphere = 4
# mesh = pv.Icosphere(nsub=5, radius=R_m) # pv.Sphere(radius=R_m, theta_resolution=res_, phi_resolution=res_, end_theta=360)
mesh = pv.Cube(x_length=2, y_length=2, z_length=2)
mesh = mesh.triangulate().clean()
mesh = mesh.subdivide(3, subfilter='linear').clean()
# mesh.rotate_x(-45, inplace=True)
mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
t0 = time.time()
r_inout = np.array([1, 0, 0])
r_inout = r_inout / np.linalg.norm(r_inout)
res_prop = compute_ray_tracing_fast(mesh, r_inout, res_x, res_y)
print("time ms:", (time.time() - t0) * 1000)

show_ray_tracing_fast(mesh, res_prop, filename="", show_mesh=True, save_3d=False)


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
A_surf = -px_area / cos_th                   # (N,)
# Projected area to the beam (used in SRP closed-form): A_proj = A_surf * cos(theta) = px_area
A_proj = px_area                            # (scalar)

# Build tangent unit vector (orthonormal basis {n, t_hat}) consistent with ray_dir
ref = np.cross(ray_dir, normal_cell)                  # (N,3)
ref_norm = np.linalg.norm(ref, axis=1, keepdims=True)
ref_norm[ref_norm == 0.0] = 1.0
ref = ref / ref_norm
t_hat = np.cross(normal_cell, ref)
t_hat = t_hat / np.linalg.norm(t_hat, axis=1, keepdims=True)

# === Physical parameters (SI) ===
kB     = 1.3806488e-23
T_wall = 303.15               # K (30 C)
v_mag  = 7800.0               # m/s
rho    = 5e-13                # kg/m^3 (example)
q_vel  = 0.5 * rho * v_mag**2
MM      = 18e-3               # kg/mol
NA      = 6.02214076e23       # 1/mol
m_particle = MM / NA          # kg
vm = np.sqrt(2.0 * kB * T_wall / m_particle)
s_ = v_mag / vm               # speed ratio (scalar)

# === Free-molecular drag model (vectorized) ===
diffuse_drag = 0.05
S  = s_ * cos_th
St = s_ * np.sqrt(np.maximum(0.0, 1.0 - cos_th**2))
Pi  = S * np.exp(-(S**2)) + np.sqrt(np.pi) * (S**2 + 0.5) * (1.0 + erf(S))
Chi = np.exp(-(S**2)) + np.sqrt(np.pi) * S * (1.0 + erf(S))

term1 = ((2.0 - diffuse_drag) / np.sqrt(np.pi)) * Pi / (s_**2)
term2 = (diffuse_drag / 2.0) * Chi / (s_**2) * np.sqrt(T_wall / (T_wall + 0.0))
cn = term1 + term2                               # normal coefficient per hit (N)
ct = diffuse_drag * St * Chi / (np.sqrt(np.pi) * s_**2)

# Drag force per hit on the surface patch:
# F_d = q * A_surf * ( cn * n + ct * t_hat )
# Equation 11.24
F_d = q_vel * A_proj * (cn[:,None] * (-normal_cell) + ct[:,None] * t_hat)   # (N,3)
T_d = np.cross(r_m, F_d)                                           # (N,3)

# === Sums ===
F_drag_total = F_d.sum(axis=0)
T_drag_total = T_d.sum(axis=0)

v_vec = ray_dir
F_drag_ref = q_vel * v_vec * A_proj_2d_sphere
rel_err_drag = np.linalg.norm(F_drag_total - F_drag_ref)/np.linalg.norm(F_drag_ref) * 100

print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)
print(f"Drag Force [N]:")
print(f"  Ray-traced:  {F_drag_total}")
print(f"  Analytical:  {F_drag_ref}")
print(f"  Error: {rel_err_drag:.3f}%")
