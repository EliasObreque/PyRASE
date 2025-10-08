"""
Created by Elias Obreque
Date: 30/09/2025
email: els.obrq@gmail.com
"""
import time
import numpy as np
import pyvista as pv
from core.optimal_ray_tracing import compute_ray_tracing_fast_optimized
from core.srp_models import compute_srp_lambert_model, compute_spherical_srp_model
from rich import print


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
# warm code
res_prop = compute_ray_tracing_fast_optimized(mesh, input_vector[0], 10, 10)

for i in range(3):
    r_inout = input_vector[i]
    r_inout = r_inout / np.linalg.norm(r_inout)
    t0 = time.time()
    res_prop = compute_ray_tracing_fast_optimized(mesh, r_inout, res_x, res_y)
    print(f"[yellow]Time ms:  {(time.time() - t0) * 1000}[/yellow]")

    #plot_normals_with_glyph(mesh, res_prop, arrow_scale=0.01)
    #show_ray_tracing_fast(mesh, res_prop, filename="", show_mesh=True, save_3d=False)

    # === Physical parameters (SI) ===
    # Cr     = 1.12
    diffuse = diffuse_list[i]
    spec = specular_list[i]
    absor = 1 - diffuse - spec
    Cr = 1 + spec + 2*diffuse/3
    com_m = mesh.center

    F_srp_total, T_srp_total, F_s, T_s = compute_srp_lambert_model(res_prop, r_inout, com_m, diffuse, spec)

    # plot_force_torque_heatmaps(res_prop, r_m, "Position", f"results/position_case_{i + 1}.png")
    #
    # plot_force_torque_heatmaps(res_prop, F_s, "Force", f"results/force_case_{i + 1}.png")
    #
    # plot_force_torque_heatmaps(res_prop, T_s, "Torque", f"results/torque_s_case_{i + 1}.png")

    #plot_torque_heatmaps(res_prop, T_s,
    #                     filename=f"torque_case_{i + 1}.png")

    F_srp_cb = compute_spherical_srp_model(r_inout, Cr, A_proj_2d_sphere)
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
