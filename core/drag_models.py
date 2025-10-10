"""
Created by Elias Obreque
Date: 07/10/2025
email: els.obrq@gmail.com
"""

import numpy as np
from scipy.special import erf
from pyatmos import download_sw_nrlmsise00,read_sw_nrlmsise00
from pyatmos import nrlmsise00

swfile = download_sw_nrlmsise00()

swdata = read_sw_nrlmsise00(swfile)

# === Physical parameters (SI) ===
kB     = 1.3806488e-23 # J/K
NA = 6.02214076e23  # 1/mol            # kg/mol
# Gas properties (atomic oxygen dominant at 200 km)

# Molecular masses (g/mol)
M_species = {
    'He': 4.0026,
    'O': 15.999,      # Atomic oxygen
    'N2': 28.0134,
    'O2': 31.9988,
    'Ar': 39.948,
    'H': 1.00794,
    'N': 14.0067,     # Atomic nitrogen
}


def get_atmospheric_condition(t: str, alt, lat=0.0, lon=0.0):
    output = nrlmsise00(t, (lat, lon, alt), swdata)
    T = output.T      # Temperature at altitude
    rho = output.rho  # Total mass density (kg/m³)

    # Access the number densities from the output
    n_densities = {
        'He': output.nd['He'],
        'O': output.nd['O'],
        'N2': output.nd['N2'],
        'O2': output.nd['O2'],
        'Ar': output.nd['Ar'],
        'H': output.nd['H'],
        'N': output.nd['N'],
    }

    # Calculate total number density
    n_total = sum(n_densities.values())
    # Calculate mean molecular mass (weighted by number density)
    m_mean = sum(n_densities[species] * M_species[species]
                 for species in n_densities.keys()) / n_total
    # Convert to kg/mol for gas constant calculation
    m_mean_kg = m_mean / 1000
    # Calculate specific gas constant for this altitude
    r_specific = 8314.46 / m_mean  # J/(kg·K)

    m_particle = m_mean_kg / NA  # kg
    return T, rho, m_particle, r_specific


def get_tangential_vector(ray_dir, normal_cell):
    # Build tangent unit vector (orthonormal basis {n, t_hat}) consistent with ray_dir
    ref = np.cross(ray_dir, normal_cell)  # (N,3)
    ref_norm = np.linalg.norm(ref, axis=1, keepdims=True)
    ref_norm[ref_norm == 0.0] = 1.0
    ref = ref / ref_norm
    t_hat = np.cross(normal_cell, ref)
    t_norm = np.linalg.norm(t_hat, axis=1, keepdims=True)
    t_norm[t_norm == 0.0] = 1.0
    t_hat = t_hat / t_norm
    return t_hat

def compute_fmf_drag_model(q_vel, ray_dir, normal_cell, hits_pos, area_proj, cn, ct,
                           com_m=np.zeros(3)):

    t_hat = get_tangential_vector(ray_dir, normal_cell)

    # Drag force per hit on the surface patch:
    # F_d = q * A_surf * ( cn * n + ct * t_hat )
    # Equation 11.24

    r_m = hits_pos - com_m
    f_d = q_vel * area_proj[:, None] * (cn[:, None] * (-normal_cell) + ct[:, None] * t_hat)  # (N,3)
    t_d = np.cross(r_m, f_d)  # (N,3)

    # === Sums ===
    f_drag_total = f_d.sum(axis=0)
    t_drag_total = t_d.sum(axis=0)
    return f_drag_total, t_drag_total, f_d, t_d

def compute_coefficients_schaaf(cos_th, v_mag, sigma_d, sigma_t, temp_inf, temp_wall, m_particle):
    # Schaaf and Chambre
    cos_th = np.abs(cos_th)
    vm = np.sqrt(2.0 * kB * temp_inf / m_particle)
    s_ = v_mag / vm  # speed ratio (scalar)
    # === Free-molecular drag model (vectorized) ===
    s_n = s_ * cos_th
    s_t = s_ * np.sqrt(np.maximum(0.0, 1.0 - cos_th ** 2))
    Pi = s_n * np.exp(-(s_n ** 2)) + np.sqrt(np.pi) * (s_n ** 2 + 0.5) * (1.0 + erf(s_n))
    Chi = np.exp(-(s_n ** 2)) + np.sqrt(np.pi) * s_n * (1.0 + erf(s_n))

    term1 = ((2.0 - sigma_d) / np.sqrt(np.pi)) * Pi / (s_ ** 2)
    term2 = (sigma_d / 2.0) * Chi / (s_ ** 2) * np.sqrt(temp_wall / temp_inf)
    cn = term1 + term2  # normal coefficient per hit (N)
    ct = sigma_t * s_t * Chi / (np.sqrt(np.pi) * s_ ** 2)
    return cn, ct

def compute_sphere_drag_model(q_vel, v_vec, a_proj_2d_sphere):
    return q_vel * v_vec * a_proj_2d_sphere


def compute_analytical_prism_coefficients(lx, ly, lz, alpha, beta, v_inf, sigma_n, sigma_t,
                                          temp_inf, temp_wall, m_particle, A_ref=None):
    """Analytical solution for rectangular prism"""
    if A_ref is None:
        A_ref = ly * lz

    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    v_inf_vec = np.array([v_inf, 0, 0])

    # Rotation matrices
    Ry = np.array([
        [np.cos(alpha_rad), 0, np.sin(alpha_rad)],
        [0, 1, 0],
        [-np.sin(alpha_rad), 0, np.cos(alpha_rad)]
    ])
    Rz = np.array([
        [np.cos(beta_rad), -np.sin(beta_rad), 0],
        [np.sin(beta_rad), np.cos(beta_rad), 0],
        [0, 0, 1]
    ])
    R_body_to_inertial = Rz @ Ry
    v_body = R_body_to_inertial.T @ v_inf_vec
    v_body_unit = v_body / np.linalg.norm(v_body)

    # 6 faces
    faces = {
        'x_pos': {'normal': np.array([1, 0, 0]), 'area': ly * lz},
        'x_neg': {'normal': np.array([-1, 0, 0]), 'area': ly * lz},
        'y_pos': {'normal': np.array([0, 1, 0]), 'area': lx * lz},
        'y_neg': {'normal': np.array([0, -1, 0]), 'area': lx * lz},
        'z_pos': {'normal': np.array([0, 0, 1]), 'area': lx * ly},
        'z_neg': {'normal': np.array([0, 0, -1]), 'area': lx * ly},
    }

    F_total_body = np.zeros(3)

    for face_name, face_data in faces.items():
        normal = face_data['normal']
        area = face_data['area']
        cos_theta = np.dot(v_body_unit, normal)

        if cos_theta > 0:
            cn, ct = compute_coefficients_schaaf(cos_theta, v_inf, sigma_n, sigma_t,
                                                 temp_inf, temp_wall, m_particle)
            v_parallel = v_body_unit - cos_theta * normal
            v_parallel_mag = np.linalg.norm(v_parallel)

            if v_parallel_mag > 1e-10:
                tangent = v_parallel / v_parallel_mag
            else:
                tangent = np.zeros(3)

            F_normal = cn * area * normal
            F_tangential = ct * area * tangent
            F_total_body += F_normal + F_tangential

    C_A = F_total_body[0] / A_ref
    C_S = F_total_body[1] / A_ref
    C_N = F_total_body[2] / A_ref
    Cd = np.linalg.norm(F_total_body) / A_ref

    return C_A, C_S, C_N, Cd

# ============================================================================
# ANALYTICAL SOLUTION - Analytic Free Molecular Aerodynamics  for Rapid Propagation of Resident Space Objects
# ============================================================================
def compute_sentman_coefficients(cos_theta, s, alpha_E, T_w, T_inf):
    """
    Sentman model  C_p (pressure) and C_tau (shear)

    Parameters:
    -----------
    cos_theta : np.ndarray
        Coseno del ángulo de incidencia
    s : float
        Speed ratio = v_inf / v_m
    alpha_E : float
        Energy accommodation coefficient (0 a 1)
        α = 1.0 → diffuse reflection (complete accommodation)
        α = 0.0 → specular reflection (no accommodation)
    T_w : float
        Wall temperature [K]
    T_inf : float
        Free stream temperature [K]
    """
    sqrt_pi = np.sqrt(np.pi)
    cos_theta = np.abs(cos_theta)

    exp_term = np.exp(-s ** 2 * cos_theta ** 2)
    erf_term = erf(s * cos_theta)

    # C_p (Pressure coefficient) - Ecuación (6) del paper Sentman
    term1 = (cos_theta ** 2 + 1 / (2 * s ** 2)) * (1 + erf_term)

    term2 = (cos_theta / (s * sqrt_pi)) * exp_term

    term3 = 0.5 * np.sqrt(2 / 3) * (1 + alpha_E * (T_w / T_inf - 1))

    term4 = sqrt_pi * cos_theta * (1 + erf_term) + (1 / s) * exp_term

    C_p = term1 + term2 + term3 * term4

    sin_theta = np.sqrt(np.maximum(0, 1 - cos_theta ** 2))

    C_tau = sin_theta * cos_theta * (
            (1 + erf_term) +
            (sin_theta / (s * sqrt_pi)) * exp_term
    )

    return C_p, C_tau


def compute_sphere_drag_sentman(s, alpha_E, T_w, T_inf):
    """
    Sentman model for sphere drag coefficient in free-molecular flow

    Parameters:
    -----------
    s : float
        Speed ratio = V_inf / v_m where v_m = sqrt(2*R*T_inf)
    alpha_E : float
        Energy accommodation coefficient (0 to 1)
    T_w : float
        Wall temperature [K]
    T_inf : float
        Freestream temperature [K]

    Returns:
    --------
    C_D : float
        Drag coefficient
    """
    sqrt_pi = np.sqrt(np.pi)

    # Incident molecular flux contribution
    exp_term = np.exp(-s ** 2)
    erf_term = erf(s)

    C_D_incident = (2 / (sqrt_pi * s ** 3)) * (
            exp_term + s * sqrt_pi * (0.5 + s ** 2) * erf_term
    )

    # Reflected/re-emitted contribution with accommodation
    C_D_reflected = np.sqrt(np.pi / 2) * (1 / s ** 2) * (
            1 + alpha_E * (T_w / T_inf - 1)
    )

    C_D = C_D_incident + C_D_reflected

    return C_D


def get_sphere_drag_coefficient(s, T_ratio, sigma_N, sigma_T):
    """
    CLL drag coefficient for a sphere in free molecular flow

    Parameters:
    -----------
    s : float
        Speed ratio (v_inf / v_m)
    T_ratio : float
        Temperature ratio (T_wall / T_inf)
    sigma_N : float
        Normal momentum accommodation coefficient (0 to 1)
    sigma_T : float
        Tangential momentum accommodation coefficient (0 to 1)

    Returns:
    --------
    Cd : float
        Drag coefficient

    Reference:
    ----------
    Your provided equation - CLL model for sphere
    """
    s2 = s ** 2
    s3 = s ** 3
    s4 = s ** 4

    exp_s2 = np.exp(-s2)
    sqrt_pi = np.sqrt(np.pi)
    erf_s = erf(s)
    sqrt_Tw_Tinf = np.sqrt(T_ratio)

    # First line of the equation
    term1 = (2.0 - sigma_N + sigma_T) * (
            0.5 +
            0.5 * erf_s * (1.0 + 1.0 / s2 - 1.0 / (4.0 * s4)) +
            ((1.0 + 2.0 * s2) * exp_s2) / (4.0 * s3 * sqrt_pi)
    )

    # Second line of the equation
    term2 = (sigma_N / (3.0 * s)) * sqrt_Tw_Tinf * (1.0 + erf_s)

    term3 = (2.0 - sigma_N) / (2.0 * s2)

    term4 = (sigma_N / (6.0 * s4)) * (1.0 + (2.0 * s2 - 1.0) * exp_s2) * sqrt_Tw_Tinf

    Cd = term1 + term2 + term3 + term4

    return Cd

if __name__ == '__main__':
    pass
