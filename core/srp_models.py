"""
Created by Elias Obreque
Date: 07/10/2025
email: els.obrq@gmail.com
"""
import numpy as np


def compute_srp_lambert_model(res_prop, r_inout, com_m, diffuse, spec, p_srp=4.56e-6):
    # === Ray-tracing outputs (already filtered) ===
    hits     = res_prop['hit_points']           # (N,3), mm
    px_w     = res_prop['pixel_width']          # mm
    px_h     = res_prop['pixel_height']         # mm
    ray_dir  = r_inout         #
    cos_th   = res_prop['cos_th']
    normal_cell = res_prop['cell_normal']

    px_area = (px_w * px_h)
    # Lever arm for torque (about mesh center)
    r_m   = hits - com_m                      # (N,3)

    # Projected area assignment per-hit.
    A_proj = px_area                            # (scalar)

    # === SRP (Lambert + specular ideal simplified) ===
    # Using projected area to the beam (constant per hit in this setup): A_proj = px_area
    # Force per hit: F_s = P_srp * A_proj * [ diffuse * d + 2(spec*cos + (diffuse/3)) * n ]
    F_s = p_srp * A_proj * ((1 - spec) * ray_dir + (2.0*(spec*cos_th - diffuse/3.0))[:,None] * normal_cell)
    T_s = np.cross(r_m, F_s)

    # === Sums ===
    F_srp_total  = F_s.sum(axis=0)
    T_srp_total  = T_s.sum(axis=0)
    return F_srp_total, T_srp_total, F_s, T_s


def compute_spherical_srp_model(r_inout, cr, a_proj_2d_sphere, p_srp=4.56e-6):
    return p_srp * a_proj_2d_sphere * r_inout * cr

if __name__ == '__main__':
    pass
