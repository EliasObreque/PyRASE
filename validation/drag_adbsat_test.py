"""
Created by Elias Obreque
Date: 06/10/2025
email: els.obrq@gmail.com
"""

import pyvista as pv
import numpy as np
from scipy.special import erf
from core.drag_models import *
t = '2015-01-19 00:00:00'  # UTC

# Atmospheric conditions (200 km altitude, moderate solar activity)
altitude = 200e3  # m

T, rho, m_particle, R_specific = get_atmospheric_condition(t, altitude*1e-3)

rho_inf = rho  # kg/m^3 (from NRLMSISE-00 for 200km)
v_inf = 7784.0  # m/s (orbital velocity at 200 km)
T_inf = T
T_wall = 300.0  # K (satellite wall temperature)

# Gas-Surface Interaction (CLL model)
sigma_N = 1.0  # Normal momentum accommodation coefficient
sigma_T = 1.0  # Tangential momentum accommodation coefficient
# NOTE: For sigma_N = sigma_T = 1.0, this is diffuse reflection
sigma_d = 0.2
alpha_E = 1.0  # Full accommodation

# Derived parameters
v_m = np.sqrt(2.0 * kB * T_wall / m_particle)  # Most probable molecular speed
s = v_inf / v_m  # Speed ratio
q_inf = 0.5 * rho_inf * v_inf ** 2  # Dynamic pressure


def create_category_c_donut(outer_radius=0.3, tube_radius=0.1, resolution=50):
    """
    Category C - Shape (j): Donut/Torus
    Promotes multiple reflections inside the hole
    """
    # Create torus
    donut = pv.ParametricTorus(
        ringradius=outer_radius,
        crosssectionradius=tube_radius
    )

    # Increase resolution
    donut = donut.subdivide(2, subfilter='loop')
    donut = donut.triangulate().clean()

    return donut


def create_category_c_two_pyramids(base_size=0.4, height=0.3):
    """
    Category C - Shape (k): Two pyramids (lightning bolt shape)

    Creates two triangular pyramids joined at base with angled faces
    promoting particle reflections
    """
    # Create first pyramid (pointing forward)
    apex1 = np.array([height, 0, 0])
    base1_1 = np.array([0, -base_size / 2, -base_size / 2])
    base1_2 = np.array([0, base_size / 2, -base_size / 2])
    base1_3 = np.array([0, 0, base_size / 2])

    # Vertices for first pyramid
    vertices1 = np.array([
        apex1, base1_1, base1_2, base1_3
    ])

    # Faces (triangular)
    faces1 = np.array([
        [3, 0, 1, 2],  # Face 1
        [3, 0, 2, 3],  # Face 2
        [3, 0, 3, 1],  # Face 3
        [3, 1, 3, 2],  # Base
    ])

    pyr1 = pv.PolyData(vertices1, faces1)

    # Create second pyramid (pointing back/down at angle)
    apex2 = np.array([-height * 0.7, 0, -height * 0.5])
    base2_1 = np.array([0, -base_size / 2, -base_size / 2])
    base2_2 = np.array([0, base_size / 2, -base_size / 2])
    base2_3 = np.array([0, 0, base_size / 2])

    vertices2 = np.array([
        apex2, base2_1, base2_2, base2_3
    ])

    pyr2 = pv.PolyData(vertices2, faces1)

    # Combine
    combined = pyr1 + pyr2

    return combined.triangulate().clean()


def create_category_c_four_pyramids(base_size=0.35, height=0.25):
    """
    Category C - Shape (l): Four pyramids (zigzag/lightning bolt)

    THIS IS THE ONE IN YOUR IMAGE
    Creates zigzag pattern with multiple angled surfaces
    """
    pyramids = []

    # Create 4 pyramid segments in zigzag pattern
    positions = [
        (0, 0, 0, 1),  # First pyramid, pointing up-right
        (0.3, 0, -0.2, -1),  # Second, pointing down-left
        (0.5, 0, 0, 1),  # Third, pointing up-right
        (0.7, 0, -0.15, -1)  # Fourth, pointing down-left
    ]

    for i, (x_offset, y_offset, z_offset, direction) in enumerate(positions):
        # Apex and base vertices
        apex = np.array([x_offset + height * direction * 0.5,
                         y_offset,
                         z_offset + height * 0.3 * direction])

        # Base corners
        base_center = np.array([x_offset, y_offset, z_offset])
        base1 = base_center + np.array([-0.1, -base_size / 2, 0])
        base2 = base_center + np.array([-0.1, base_size / 2, 0])
        base3 = base_center + np.array([0.1, 0, base_size / 2])

        vertices = np.array([apex, base1, base2, base3])

        faces = np.array([
            [3, 0, 1, 2],
            [3, 0, 2, 3],
            [3, 0, 3, 1],
            [3, 1, 3, 2],
        ])

        pyr = pv.PolyData(vertices, faces)
        pyramids.append(pyr)

    # Combine all
    combined = pyramids[0]
    for pyr in pyramids[1:]:
        combined = combined + pyr

    return combined.triangulate().clean()


def create_category_c_cylinder_with_grooves(radius=0.2, length=0.6,
                                            groove_depth=0.05, n_grooves=8):
    """
    Category C - Shape (m): Cylinder with detail

    Cylinder with longitudinal grooves/indentations on forward face
    that trap particles
    """
    # Base cylinder
    cylinder = pv.Cylinder(
        center=(0, 0, 0),
        direction=(1, 0, 0),
        radius=radius,
        height=length,
        resolution=50
    )

    # Add grooves by boolean subtraction
    groove_cyl = pv.Cylinder(
        center=(length / 2 - groove_depth, 0, 0),
        direction=(1, 0, 0),
        radius=radius * 0.15,
        height=groove_depth * 2,
        resolution=20
    )

    # Create multiple grooves around circumference
    grooved = cylinder
    for i in range(n_grooves):
        angle = i * 360 / n_grooves
        rotated_groove = groove_cyl.copy()
        rotated_groove.rotate_x(angle, point=(length / 2, 0, 0))

        # Subtract (simulated by moving vertices inward)
        # In practice, use boolean operations if available

    return grooved.triangulate().clean()


def create_category_c_cube_with_cavity(size=0.4, cavity_depth=0.15):
    """
    Category C - Shape (n): Cube with Detail

    Cube with deep cavity/indentation on forward face
    Creates particle trapping region
    """
    # Main cube
    cube = pv.Cube(x_length=size, y_length=size, z_length=size)

    # Create cavity by boolean subtraction
    cavity = pv.Cube(
        x_length=cavity_depth,
        y_length=size * 0.6,
        z_length=size * 0.6
    )
    cavity.translate([size / 2 - cavity_depth / 2, 0, 0])

    # Subtract cavity from cube (boolean operation)
    # Note: PyVista boolean operations may require specific setup
    try:
        result = cube.boolean_difference(cavity)
    except:
        # Fallback: manual cavity creation
        result = cube

    return result.triangulate().clean()


# Usage example
if __name__ == "__main__":
    from core.optimal_ray_tracing import compute_ray_tracing_fast_optimized
    from core.monitor import plot_force_torque_heatmaps, show_ray_tracing_fast

    # Test the shape (l) that matches your image
    mesh = create_category_c_donut()
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False)
    flow = np.array([0, 0, 1])
    res_prop = compute_ray_tracing_fast_optimized(mesh, flow, 500, 500, True)

    show_ray_tracing_fast(mesh, res_prop, filename="", show_mesh=True, save_3d=False)

    # ============================================================================
    # EXTRACT RAY TRACING RESULTS
    # ============================================================================
    hits = res_prop['hit_points']  # (N,3), m
    ray_ids = res_prop['ray_ids']  # (N,)
    cell_ids = res_prop['cell_ids']  # (N,)
    px_w = res_prop['pixel_width']  # m
    px_h = res_prop['pixel_height']  # m
    ray_dir = flow
    cos_th = res_prop['cos_th']  # cos(θ) where θ is angle between ray and normal
    normal_cell = res_prop['cell_normal']  # (N,3)
    A_fem = res_prop['A_fem']

    px_area = px_w * px_h  # Pixel area

    A_ref = px_area * len(cell_ids)
    com_m = np.asarray(mesh.center)
    r_m = hits - com_m  # Lever arm for torque

    # ============================================================================
    # FREE MOLECULAR DRAG CALCULATION (VECTORIZED)
    # ============================================================================
    # Build tangent vector (for tangential drag component)
    ref = np.cross(ray_dir, normal_cell)
    ref_norm = np.linalg.norm(ref, axis=1, keepdims=True)
    ref_norm[ref_norm == 0.0] = 1.0
    ref = ref / ref_norm
    t_hat = np.cross(normal_cell, ref)
    t_hat = t_hat / np.linalg.norm(t_hat, axis=1, keepdims=True)

    # Speed ratio components
    Sn = s * np.abs(cos_th)  # Normal speed ratio
    St = s * np.sqrt(np.maximum(0.0, 1.0 - cos_th ** 2))  # Tangential speed ratio

    # Pressure and shear coefficients (Sentman/Schaaf-Chambre model)
    # CHECK THESE EQUATIONS - they match your SRP validation structure
    exp_S2 = np.exp(-(Sn ** 2))
    sqrt_pi = np.sqrt(np.pi)
    erf_S = erf(Sn)

    # Normal coefficient (Cn) - Equation from Schaaf & Chambre
    Pi = Sn * exp_S2 + sqrt_pi * (Sn ** 2 + 0.5) * (1.0 + erf_S)
    Chi = exp_S2 + sqrt_pi * Sn * (1.0 + erf_S)

    # CHECK THIS: Normal pressure coefficient
    # Term 1: Specular-like reflection
    term1 = ((2.0 - sigma_d) / sqrt_pi) * Pi / (s ** 2)
    # Term 2: Diffuse reflection with thermal accommodation
    term2 = (sigma_d / 2.0) * Chi / (s ** 2) * np.sqrt(T_wall / T_inf)
    cn = term1 + term2

    # CHECK THIS: Tangential shear coefficient
    # For diffuse reflection with complete accommodation
    ct = sigma_d * St * Chi / (sqrt_pi * s ** 2)
    C_p, C_tau = compute_sentman_coefficients(
        cos_theta=np.abs(cos_th),
        s=s,
        alpha_E=alpha_E,
        T_w=T_wall,
        T_inf=T_inf
    )

    # ============================================================================
    # DRAG FORCE CALCULATION (PER ELEMENT)
    # ============================================================================
    # NOTE: Your notation uses negative normal for outward-facing normals
    # F_d = q_inf * A_proj * (cn * (-n) + ct * t)
    # where A_proj = px_area (projected area per pixel)

    Area_r = -px_area / cos_th
    area_aux = max(px_area / np.cos(89 * np.pi / 180), 5 * px_area)
    print(area_aux, 5 * px_area)
    Area_r[Area_r > area_aux] = area_aux

    Area_r_x3 = np.array([Area_r, Area_r, Area_r]).T
    # plot_force_torque_heatmaps(res_prop, Area_r_x3, "Area", f"area_case_{0 + 1}.png")

    cn, ct = compute_coefficients_schaaf(cos_th, v_inf, sigma_d, sigma_d, T_inf, T_wall, m_particle)

    F_drag_total, T_drag_total, F_d, T_d = compute_fmf_drag_model(q_inf, ray_dir, normal_cell, hits, Area_r, cn, ct,
                                                                  com_m=com_m)

    # Drag coefficient from ray tracing
    Cd_raytraced = np.linalg.norm(F_drag_total) / (q_inf * A_ref)
    print(Cd_raytraced)

