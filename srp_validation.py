"""
Created by Elias Obreque
Date: 30/09/2025
email: els.obrq@gmail.com
"""
import numpy as np
import pyvista as pv
from src.monitor import show_ray_tracing_fast, plot_normals_with_glyph
from src.ray_tracing import compute_ray_tracing_fast
import time


true_values = [np.array([-2.783188, 0, 0]) * 1e-5,
               np.array([-3.267220, 0, 0]) * 1e-5,
               np.array([-2.157385, -2.157385, 0]) * 1e-5]

diffuse_list = [0.8, 0.0, 0.4]
specular_list = [0.0, 0.8, 0.4]
input_vector = [np.array([-1, 0, 0]),
                np.array([-1, 0, 0]),
                np.array([-1, -1, 0])]

res_ = 50
res_x = res_y = 500
A_proj_2d_sphere = 4.0
P_srp = 4.56e-6  # N/m^2 (solar radiation pressure at 1 AU)

# mesh = pv.Icosphere(nsub=5, radius=R_m) # pv.Sphere(radius=R_m, theta_resolution=res_, phi_resolution=res_, end_theta=360)
mesh = pv.Cube(x_length=2, y_length=2, z_length=2)
mesh = mesh.triangulate().clean()
mesh = mesh.subdivide(2, subfilter='linear').clean()
mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)

for i in range(3):
    r_inout = input_vector[i]
    r_inout = r_inout / np.linalg.norm(r_inout)
    t0 = time.time()
    res_prop = compute_ray_tracing_fast(mesh, r_inout, res_x, res_y)
    print("time ms:", (time.time() - t0) * 1000)

    plot_normals_with_glyph(mesh, res_prop, arrow_scale=0.01)
    show_ray_tracing_fast(mesh, res_prop, filename="", show_mesh=True, save_3d=False)

    # === Ray-tracing outputs (already filtered) ===
    hits     = res_prop['hit_points']           # (N,3), mm
    ray_ids  = res_prop['ray_ids']              # (N,)
    cell_ids = res_prop['cell_ids']             # (N,) or None
    px_w     = res_prop['pixel_width']          # mm
    px_h     = res_prop['pixel_height']         # mm
    ray_dir  = r_inout         #
    cos_th   = res_prop['cos_th']
    normal_cell = res_prop['cell_normal']

    px_area = (px_w * px_h)
    # Lever arm for torque (about mesh center)
    com_m = np.asarray(mesh.center)
    r_m   = hits - com_m                      # (N,3)

    # Projected area assignment per-hit.
    A_proj = px_area                            # (scalar)

    # === Physical parameters (SI) ===
    # Cr     = 1.12
    diffuse = diffuse_list[i]
    spec    = specular_list[i]
    absor = 1 - diffuse - spec
    Cr = 1 + spec + 2*diffuse/3

    # === SRP (Lambert + specular ideal simplified) ===
    # Using projected area to the beam (constant per hit in this setup): A_proj = px_area
    # Force per hit: F_s = P_srp * A_proj * [ diffuse * d + 2(spec*cos + (diffuse/3)) * n ]
    F_s = P_srp * A_proj * ((1 - spec) * ray_dir + (2.0*(spec*cos_th - diffuse/3.0))[:,None] * normal_cell)
    T_s = np.cross(r_m, F_s)

    # === Sums ===
    F_srp_total  = F_s.sum(axis=0)
    T_srp_total  = T_s.sum(axis=0)

    # plot_force_torque_heatmaps(res_prop, r_m, "Position", f"results/position_case_{i + 1}.png")
    #
    # plot_force_torque_heatmaps(res_prop, F_s, "Force", f"results/force_case_{i + 1}.png")
    #
    # plot_force_torque_heatmaps(res_prop, T_s, "Torque", f"results/torque_s_case_{i + 1}.png")

    #plot_torque_heatmaps(res_prop, T_s,
    #                     filename=f"torque_case_{i + 1}.png")

    F_srp_cb = P_srp * A_proj_2d_sphere *ray_dir * Cr
    F_srp_ref = true_values[i]

    rel_err_srp_fem = np.linalg.norm(F_srp_total - F_srp_ref)/np.linalg.norm(F_srp_ref) * 100
    rel_err_srp_cb = np.linalg.norm(F_srp_cb - F_srp_ref) / np.linalg.norm(F_srp_ref) * 100

    print("=" * 60)
    print("VALIDATION RESULTS - Sphere")
    print(f"  True value:  {F_srp_ref}")
    print("=" * 60)
    print(f"SRP Force [N]:")
    print(f"  Ray-traced:  {F_srp_total}")
    print(f"  Cannonball model (CR={Cr}):  {F_srp_cb}")
    print(f"  Ray-traced Torque:  {T_srp_total}")

    print(f"  FEM - Error: {rel_err_srp_fem:.3f}%")
    print(f"  CB - Error: {rel_err_srp_cb:.3f}%")
