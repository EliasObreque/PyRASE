"""
Sphere Drag Validation for Free Molecular Flow
Created by Elias Obreque (with validation equations)
Date: 02/10/2025
email: els.obrq@gmail.com

Validation against Sentman analytical solution for sphere in FMF
Reference: Sentman (1961), "Free Molecule Flow Theory"
"""
import time
import numpy as np
from scipy.special import erf
import pyvista as pv
from rich import print

from core.optimal_ray_tracing import compute_ray_tracing_fast_optimized
from core.monitor import plot_force_torque_heatmaps, show_ray_tracing_fast
from core.drag_models import *

mu = 398600.4418  # km³/s² (Earth's gravitational parameter)
R_earth = 6371  # km (mean radius)

# ============================================================================
# TEST CASE PARAMETERS (Following ADBSat validation)
# ============================================================================
# Geometry

R_sphere = 1.0  # m (sphere radius)
A_ref = np.pi * R_sphere ** 2  # Reference area (cross-section)

print("="*70)
print("TESTING NRLMSISE-00 AT DIFFERENT ALTITUDES")
print("="*70)

t = '2015-01-19 00:00:00' # time(UTC)

altitudes = [100, 200, 250, 300, 400, 500, 600]

print(f"\n{'Alt (km)':<10} {'T (K)':<15} {'Density (kg/m³)':<20} {'MM (g)':<15} {'R* (J/(kg·K))':<15}")
print("-"*45)

for alt in altitudes:
    T, rho, m_particle, R_specific = get_atmospheric_condition(t, alt)
    print(f"{alt:<10} {T:<15.1f} {rho:<20.3e} {m_particle * 1000:<15.4f} {R_specific:<15.4f}")

# Atmospheric conditions (200 km altitude, moderate solar activity)
altitude = 200  # m
T, rho, m_particle, R_specific = get_atmospheric_condition(t, altitude)
rho_inf = rho  # kg/m^3 (from NRLMSISE-00 for 200km)

r = R_earth + altitude  # km
V_orbit = np.sqrt(mu / r)  # km/s


v_inf = 7784.0  # m/s (orbital velocity at 200 km)
T_inf = T  # K (atmospheric temperature at 200 km)
T_wall = 300.0  # K (satellite wall temperature)


# Gas-Surface Interaction (CLL model)
sigma_N = 1.0  # Normal momentum accommodation coefficient
sigma_T = 1.0  # Tangential momentum accommodation coefficient
# NOTE: For sigma_N = sigma_T = 1.0, this is diffuse reflection

# Derived parameters
v_m = np.sqrt(2.0 * kB * T_wall / m_particle)  # Most probable molecular speed
s = v_inf / v_m  # Speed ratio
q_inf = 0.5 * rho_inf * v_inf ** 2  # Dynamic pressure


# ============================================================================
# MESH GENERATION
# ============================================================================
res_x = res_y = 500  # Ray tracing resolution
mesh = pv.Icosphere(nsub=8, radius=R_sphere)# pv.Sphere(radius=R_sphere, theta_resolution=50, phi_resolution=50)
mesh = mesh.triangulate().clean()
mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)

# Flow direction (along +X axis)
r_inout = np.array([1, 0, 0])
r_inout = r_inout / np.linalg.norm(r_inout)

# ============================================================================
# RAY TRACING COMPUTATION
# ============================================================================
print("[yellow]Computing ray tracing...[/yellow]")
t0 = time.time()
res_prop = compute_ray_tracing_fast_optimized(mesh, r_inout, res_x, res_y)
print(f"[yellow]Time: {(time.time() - t0) * 1000:.2f} ms[/yellow]")

# Uncomment to visualize
# show_ray_tracing_fast(mesh, res_prop, filename="", show_mesh=True, save_3d=False)

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

Area_r = res_prop['area_proj']

px_area = px_w * px_h  # Pixel area
com_m = np.asarray(mesh.center)

Area_r_x3 = np.array([Area_r, Area_r, Area_r]).T
plot_force_torque_heatmaps(res_prop, Area_r_x3, "Area", f"area_case_{0 + 1}.png")

cn, ct = compute_coefficients_schaaf(cos_th, v_inf, sigma_N, sigma_T, T_inf, T_wall, m_particle)

F_drag_total, T_drag_total, F_d, T_d = compute_fmf_drag_model(q_inf, ray_dir, normal_cell, hits, Area_r, cn, ct,
                                                              com_m=com_m)


# Drag coefficient from ray tracing
Cd_raytraced = np.linalg.norm(F_drag_total) / (q_inf * A_ref)

# ============================================================================
# ANALYTICAL REFERENCE VALUES
# ============================================================================
# Sentman analytical solution for sphere
T_ratio = T_wall / T_inf
Cd_sentman = get_sphere_drag_coefficient(s, T_ratio, sigma_N, sigma_T)
# Cd_sentman = compute_sphere_drag_sentman(s, 1.0, T_wall, T_inf)

# Drag force from analytical solution
F_drag_sentman = Cd_sentman * q_inf * A_ref * ray_dir

# Relative errors
rel_err_Cd = abs(Cd_raytraced - Cd_sentman) / Cd_sentman * 100
rel_err_F = np.linalg.norm(F_drag_total - F_drag_sentman) / np.linalg.norm(F_drag_sentman) * 100

# Expected Cd range for validation (from literature)
# At s~10-12, alpha=1.0, typical Cd ≈ 2.05-2.22
Cd_expected_range = (2.05, 2.22)
C_p, C_tau = compute_sentman_coefficients(cos_theta=np.abs(cos_th), s=s, alpha_E=1, T_w=T_wall, T_inf=T_inf)

F_drag_total_sent, _, F_d_, _ = compute_fmf_drag_model(q_inf, ray_dir, normal_cell, hits, Area_r, C_p, C_tau)
Cd_raytraced_sent = np.linalg.norm(F_drag_total_sent) / (q_inf * A_ref)
rel_err_F_sent = np.linalg.norm(F_drag_total_sent - F_drag_sentman) / np.linalg.norm(F_drag_sentman) * 100
# ============================================================================
# RESULTS OUTPUT
# ============================================================================
print("\n" + "=" * 70)
print("SPHERE DRAG VALIDATION - FREE MOLECULAR FLOW")
print("=" * 70)
print("\nTest Conditions:")
print(f"  Altitude:           {altitude:.1f} km")
print(f"  Sphere radius:      {R_sphere:.2f} m")
print(f"  Reference area:     {A_ref:.4f} m²")
print(f"  Velocity:           {v_inf:.1f} m/s")
print(f"  Density:            {rho_inf:.2e} kg/m³")
print(f"  T_inf:              {T_inf:.1f} K")
print(f"  T_wall:             {T_wall:.1f} K")
print(f"  Speed ratio (s):    {s:.2f}")
print(f"  Dynamic pressure:   {q_inf:.4e} Pa")

print("\n" + "-" * 70)
print("DRAG COEFFICIENT (Cd):")
print(f"  Ray-traced (FEM):      {Cd_raytraced:.6f}")
print(f"  Ray-traced Sentman (FEM):      {Cd_raytraced:.6f}")
print(f"  Sentman analytical:    {Cd_sentman:.6f}")
print(f"  Expected range:        {Cd_expected_range[0]:.2f} - {Cd_expected_range[1]:.2f}")
print(f"  → Error:               {rel_err_Cd:.3f}%")

print("\n" + "-" * 70)
print("DRAG FORCE [N]:")
print(f"  Ray-traced:   {F_drag_total}")
print(f"  Ray-traced - Sentman:   {F_drag_total_sent}")
print(f"  Analytical:   {F_drag_sentman}")
print(f"  → Error:      {rel_err_F:.3f}%")
print(f"  → Error:      {rel_err_F_sent:.3f}%")


print("\n" + "-" * 70)
print("DRAG TORQUE [N·m]:")
print(f"  Ray-traced:   {T_drag_total}")
print(f"  (Should be ~0 for sphere due to symmetry)")

print("\n" + "=" * 70)
if rel_err_Cd < 3.0:
    print("[green]✓ VALIDATION PASSED - Error < 3%[/green]")
elif rel_err_Cd < 5.0:
    print("[yellow]⚠ VALIDATION ACCEPTABLE - Error < 5%[/yellow]")
else:
    print("[red]✗ VALIDATION FAILED - Error > 5%[/red]")
print("=" * 70)
print(f"  → Error:      {rel_err_F:.3f}%")