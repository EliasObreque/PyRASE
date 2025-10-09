"""
Created by Elias Obreque
Date: 01/10/2025
email: els.obrq@gmail.com
"""
import time
import numpy as np

import pyvista as pv
from matplotlib import pyplot as plt
from scipy.spatial.transform.rotation import Rotation
from core.monitor import show_ray_tracing_fast, show_local_coefficient_per_angle, show_error_local_coefficient_per_angle
from core.optimal_ray_tracing import compute_ray_tracing_fast_optimized
from core.drag_models import compute_fmf_drag_model, compute_coefficients_schaaf, get_atmospheric_condition, get_tangential_vector

v_inf = 7500
T_wall = 300
T_inf = 973

t = '2015-01-19 00:00:00' # time(UTC)

_, rho, m_particle, r_specific = get_atmospheric_condition(t, 250)
q_inf = 0.5 * rho * v_inf ** 2

sigma_list = [0, 0.25, 0.5, 0.75, 1.0]
aoa_list = [0, 15, 30, 45, 60, 75, 77.5, 80, 82.5, 85, 87.5, 88, 88.5, 89, 89.5, 90]
#0, 15, 30, 45, 60, 75, 77.5,
res_ = 50
res_x = res_y = 100

r_c = 10
angle_c = 36 * np.deg2rad(1.0)
length_cone = r_c / np.tan(angle_c)


A_ref = np.pi * r_c **2
C_A_list = []
C_N_list = []
C_S_list = []

r_inout = np.array([1, 0, 0])
r_inout = r_inout / np.linalg.norm(r_inout)

error_C_A_list = []
error_C_N_list = []
error_C_S_list = []

Area_list = []

for sigma in sigma_list:
    C_A_sigma = []
    C_N_sigma = []
    C_S_sigma = []
    error_C_A_sigma = []
    error_C_N_sigma = []
    error_C_S_sigma = []
    sigma_N, sigma_T = sigma, sigma
    Area_list = []
    for aoa in aoa_list:
        mesh = pv.Cone(direction=(-1, 0, 0), radius=r_c, height=length_cone, resolution=100)
        mesh = mesh.triangulate().clean()
        mesh = mesh.subdivide(3, subfilter='linear').clean()
        mesh.rotate_y(aoa, inplace=True)
        mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
        com_m = mesh.center

        t0 = time.time()
        res_prop = compute_ray_tracing_fast_optimized(mesh, r_inout, res_x, res_y)
        print("time ms:", (time.time() - t0) * 1000)

        #show_ray_tracing_fast(mesh, res_prop, filename="", show_mesh=True, save_3d=False)

        # ============================================================================
        # EXTRACT RAY TRACING RESULTS
        # ============================================================================
        hits = res_prop['hit_points']  # (N,3), m
        ray_ids = res_prop['ray_ids']  # (N,)
        cell_ids = res_prop['cell_ids']  # (N,)
        px_w = res_prop['pixel_width']  # m
        px_h = res_prop['pixel_height']  # m
        ray_dir = r_inout
        cos_th = res_prop['cos_th']  # cos(θ) where θ is angle between ray and normal
        normal_cell = res_prop['cell_normal']  # (N,3)
        # A_fem = res_prop['A_fem']
        Area_r = res_prop['area_proj']
        A_fem_proj = res_prop['A_fem_proj']

        Area_list.append(Area_r.sum())
        print(f"Angle: {aoa}", Area_list[-1])
        # area_ratio_correction = A_fem_proj / Area_r.sum()
        # Area_r *= area_ratio_correction

        cn, ct = compute_coefficients_schaaf(cos_th, v_inf, sigma_N, sigma_T, T_inf, T_wall, m_particle)

        F_drag_total, T_drag_total, F_d, T_d = compute_fmf_drag_model(q_inf, ray_dir, normal_cell,
                                                                      hits, Area_r, cn, ct,
                                                                      com_m=com_m)

        # Drag coefficient from ray tracing
        Cd_raytraced = np.linalg.norm(F_drag_total) / (q_inf * A_ref)

        # projection
        rot_ = Rotation.from_euler('y', aoa, degrees=True)
        normal_mean_vector = rot_.as_matrix() @ r_inout
        tangential_vector = get_tangential_vector(r_inout, normal_mean_vector.reshape(1, 3))[0]
        tang_norm = np.linalg.norm(tangential_vector)
        if tang_norm == 0:
            tang_norm = 1
        tangential_vector /= tang_norm
        ref_mean_vector = np.cross(r_inout, normal_mean_vector)

        C_A = F_drag_total @ normal_mean_vector / (q_inf * A_ref)
        C_N = F_drag_total @ tangential_vector / (q_inf * A_ref)
        C_S = F_drag_total @ ref_mean_vector / (q_inf * A_ref)

        print(r_inout, normal_mean_vector, tangential_vector, ref_mean_vector)
        C_A_sigma.append(C_A)
        C_N_sigma.append(C_N)
        C_S_sigma.append(C_S)
        #ERROR
        cos_th_target = np.cos(aoa * np.pi / 180)
        cn, ct = compute_coefficients_schaaf(cos_th_target, v_inf, sigma_N, sigma_T, T_inf, T_wall, m_particle)
        cs = 0

        c_t = cn + ct + cs
        error_ca = np.abs(C_A - cn)/c_t *100
        error_cn = np.abs(C_N - ct)/c_t *100
        error_cs = np.abs(C_S - cs)/c_t *100

        error_C_A_sigma.append(error_ca)
        error_C_N_sigma.append(error_cn)
        error_C_S_sigma.append(error_cs)

        print("=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        print(f"Drag Force [N]:")
        print(f"  Ray-traced:  {F_drag_total} - {Cd_raytraced}")
        print("=" * 60)
        print("Drag coefficients:")
        print(f"  Drag Coefficients:  {C_A}")
        print(f"  Lift Coefficients:  {C_N}")
        print("=" * 60)

    C_A_list.append(C_A_sigma)
    C_N_list.append(C_N_sigma)
    C_S_list.append(C_S_sigma)

    error_C_A_list.append(error_C_A_sigma)
    error_C_N_list.append(error_C_N_sigma)
    error_C_S_list.append(error_C_S_sigma)


# Analytical calculation
C_N_sigma_analytic = []
C_T_sigma_analytic = []
C_S_sigma_analytic = []

aoa_array = np.linspace(0, 90, 100)
for sigma in sigma_list:
    sigma_N, sigma_T = sigma, sigma

    cos_th = np.cos(aoa_array * np.pi / 180)
    cn, ct = compute_coefficients_schaaf(cos_th, v_inf, sigma_N, sigma_T, T_inf, T_wall, m_particle)

    C_N_sigma_analytic.append(cn)
    C_T_sigma_analytic.append(ct)
    C_S_sigma_analytic.append(np.zeros_like(cn))


# VISUALIZATION

# error_Area_list = np.abs(np.array(Area_list) - 1)/ 1 * 100
# plt.figure()
# plt.plot(aoa_list, Area_list)
# plt.show()
show_local_coefficient_per_angle(aoa_list, aoa_array,
                                 C_N_sigma_analytic, C_S_sigma_analytic, C_T_sigma_analytic,
                                 C_A_list, C_S_list, C_N_list, sigma_list, f'results/cone_aerodynamics_{res_x}.png')


show_error_local_coefficient_per_angle(aoa_list, error_C_A_list, error_C_S_list, error_C_N_list, sigma_list,
                                           f'results/error_cone_aerodynamics_res_{res_x}.png')