# -*- coding: utf-8 -*-
"""
Orbit Propagation Comparison: Ground Truth vs Spherical vs ANN Models
Created by: O Break
Date: 2025-12-16

Compares three orbit propagation methods:
1. Ground Truth: Full ray tracing with compute_ray_perturbation_step
2. Spherical Model: Classical spherical drag/SRP approximations
3. ANN Model: Neural network trained model (from ESP32 export or pickle)

Features:
- J2 perturbation included in all models
- Selective perturbation comparison (drag only, SRP only, both)
- Comprehensive error analysis and visualization
- Compatible with monitor.py plotting style
"""

import os
import pickle
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp
from tqdm import tqdm
import pyvista as pv

from core.compute_perturbation_data import compute_ray_perturbation_step
from core.optimal_ray_tracing import compute_ray_tracing_fast_optimized
from core.drag_models import spherical_drag_force, get_dynamic_pressure, compute_analytical_prism_step
from core.srp_models import spherical_srp_force
from core.ann_tools import (load_ann_models, load_ann_model,
                            ann_predict_force, ann_predict_torque, ann_predict)
from core.monitor import plot_orbit_comparison, plot_error_statistics, compute_position_error, compute_velocity_error, plot_orbit_hill_frame_alt

# ==========================
# CONSTANTS
# ==========================

# Earth parameters
MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6378137.0  # m
J2 = 1.08262668e-3  # J2 coefficient


# ==========================
# ORBIT DYNAMICS
# ==========================

def j2_acceleration(r_eci):
    """
    Compute J2 perturbation acceleration

    Parameters:
    -----------
    r_eci : np.ndarray (3,)
        Position in ECI frame [m]

    Returns:
    --------
    a_j2 : np.ndarray (3,)
        J2 acceleration [m/s^2]
    """
    r_mag = np.linalg.norm(r_eci)
    x, y, z = r_eci

    factor = -1.5 * J2 * MU_EARTH * R_EARTH ** 2 / r_mag ** 5

    a_j2 = np.array([
        factor * x * (1 - 5 * z ** 2 / r_mag ** 2),
        factor * y * (1 - 5 * z ** 2 / r_mag ** 2),
        factor * z * (3 - 5 * z ** 2 / r_mag ** 2)
    ])

    return a_j2


def sun_position_eci(jd):
    """
    Compute the Sun position vector in ECI (Earth-Centered Inertial) frame.
    
    Based on the Astronomical Almanac low-precision solar coordinates.
    
    Parameters
    ----------
    jd : float
        Julian Date
    
    Returns
    -------
    r_sun : np.ndarray
        Sun position vector in ECI [km], shape (3,)
    """
    # Julian centuries from J2000.0
    T = (jd - 2451545.0) / 36525.0

    # Mean longitude of the Sun [deg]
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2
    L0 = L0 % 360.0

    # Mean anomaly of the Sun [deg]
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2
    M_rad = np.radians(M % 360.0)

    # Equation of center [deg]
    C = (1.914602 - 0.004817 * T - 0.000014 * T**2) * np.sin(M_rad) \
      + (0.019993 - 0.000101 * T) * np.sin(2 * M_rad) \
      + 0.000289 * np.sin(3 * M_rad)

    # Sun's true longitude and true anomaly [deg]
    sun_lon = np.radians((L0 + C) % 360.0)
    nu = M + C

    # Sun-Earth distance [AU]
    e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T**2
    r_au = 1.000001018 * (1 - e**2) / (1 + e * np.cos(np.radians(nu)))

    # Obliquity of the ecliptic [deg]
    epsilon = np.radians(23.439291 - 0.0130042 * T)

    # Convert to ECI (equatorial) coordinates
    r_km = r_au * 149597870.7  # AU to km

    r_sun = r_km * np.array([
        np.cos(sun_lon),
        np.cos(epsilon) * np.sin(sun_lon),
        np.sin(epsilon) * np.sin(sun_lon)
    ])

    return r_sun

def orbital_derivatives(t, state, params):
    """
    Compute orbital state derivatives with perturbations

    Parameters:
    -----------
    t : float
        Time [s]
    state : np.ndarray (6,)
        [x, y, z, vx, vy, vz] in ECI frame
    params : dict
        {
            'model_type': 'ground_truth', 'spherical', or 'ann'
            'mesh': pv.PolyData (for ground_truth)
            'sim_data': dict
            'A_ref': float
            'mass': float
            'ann_model': torch.nn.Module (for ann)
            'ann_scaler': scaler (for ann)
            'include_drag': bool
            'include_srp': bool
            'res_x': int (for ground_truth)
            'res_y': int (for ground_truth)
        }

    Returns:
    --------
    dstate : np.ndarray (6,)
        State derivatives
    """
    r_eci = state[:3]
    v_eci = state[3:]

    r_mag = np.linalg.norm(r_eci)

    # Two-body acceleration
    a_2body = -MU_EARTH * r_eci / r_mag ** 3

    # J2 perturbation
    a_j2 = j2_acceleration(r_eci)*0

    # Perturbation acceleration
    a_pert = np.zeros(3)

    model_type = params['model_type']

    sim_data = params['sim_data']
    sim_data['alt_km'] = (r_mag - R_EARTH) / 1000
    sim_data['v_inf'] = np.linalg.norm(v_eci)
    
    jd_ = sim_data['jd']
    sun_dir = sun_position_eci(jd_)
    sun_dir /= np.linalg.norm(sun_dir)

    if sim_data['alt_km'] < 150:
        dstate = np.zeros(6)
        return dstate
    if model_type == 'ground_truth':
        # Ray tracing model
        mesh = params['mesh']
        res_x = params.get('res_x', 50)
        res_y = params.get('res_y', 50)
        mass = params['mass']

 
        if params['include_drag']:
            # Velocity in body frame (assume body = ECI for now, can add rotation if needed)
            v_body = -v_eci
            v_mag = np.linalg.norm(v_body)
            if v_mag > 1e-6:
                r_inout = v_body / v_mag

                # Compute ray tracing
                res_prop = compute_ray_tracing_fast_optimized(mesh, r_inout, res_x, res_y, verbose=0)

                # Compute forces
                F_drag_total, T_drag_total, F_srp_total, T_srp_total = compute_ray_perturbation_step(
                    res_prop, sim_data
                )
            
                a_pert += F_drag_total / mass
        if params['include_srp']:
            r_inout = sun_dir

            # Compute ray tracing
            res_prop = compute_ray_tracing_fast_optimized(mesh, r_inout, res_x, res_y, verbose=0)

            # Compute forces
            F_drag_total, T_drag_total, F_srp_total, T_srp_total = compute_ray_perturbation_step(
                res_prop, sim_data)
            a_pert += F_srp_total / mass
    elif model_type == 'ground_truth_theory':
        v_body = -v_eci
        mass = params['mass']
        if params['include_drag']:
            F_drag_total = compute_analytical_prism_step(v_body, sim_data, params)
            a_pert += F_drag_total / mass
        if params['include_srp']:
            F_srp_total = np.zeros(3)
            a_pert += F_srp_total / mass
    elif model_type == 'spherical':
        # Spherical approximation
        A_ref = params['A_ref']
        mass = params['mass']

        v_body = v_eci  # Body frame = ECI frame (simplified)

        if params['include_drag']:
            F_drag = spherical_drag_force(v_body, sim_data, A_ref, mass)
            a_pert += F_drag / mass

        if params['include_srp']:
            # Sun direction (simplified: assume constant)
            F_srp = spherical_srp_force(sun_dir, sim_data, A_ref)
            a_pert += F_srp / mass

    elif model_type == 'ann':
        # ANN model with separate force/torque models
        ann_models = params['ann_models']
        mass = params['mass']
        device = params.get('device', 'cpu')

        v_body = -v_eci  # Body frame = ECI frame (simplified)

        # Predict drag force and torque
        if params['include_drag']:
            q_ = get_dynamic_pressure(v_body, sim_data['alt_km'], sim_data['time_str'])

            scale_matrix = sim_data.get("scale_shape", np.eye(3))
            F_drag = q_ * scale_matrix @ ann_predict_force(v_body, ann_models.get('drag_f'), device)
            T_drag = q_ * scale_matrix @ ann_predict_torque(v_body, ann_models.get('drag_t'), device)

            a_pert += F_drag / mass
            # Note: Torque doesn't affect translational motion directly

        # Predict SRP force and torque
        if params['include_srp']:
            # For SRP, we need sun direction (simplified here)
            P_SRP = 4.56e-6  # N/m^2
            F_srp = P_SRP * ann_predict_force(sun_dir, ann_models.get('srp_f'), device)
            T_srp = P_SRP * ann_predict_torque(sun_dir, ann_models.get('srp_t'), device)

            a_pert += F_srp / mass
            # Note: Torque doesn't affect translational motion directly
    elif model_type == "nominal_j2":
        a_pert = np.zeros(3)
    # Total acceleration
    a_total = a_2body + a_j2 + a_pert

    # State derivatives
    dstate = np.concatenate([v_eci, a_total])

    return dstate


# ==========================
# ORBIT PROPAGATION
# ==========================

def propagate_orbit(state0, t_span, params, method='RK45', rtol=1e-9, atol=1e-12,
                    show_progress=True, desc="Propagating orbit"):
    """
    Propagate orbit with specified perturbations
    RK45, DOP853
    Parameters:
    -----------
    state0 : np.ndarray (6,)
        Initial state [x, y, z, vx, vy, vz] [m, m/s]
    t_span : tuple
        (t0, tf) in seconds
    params : dict
        Propagation parameters
    method : str
        Integration method
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    show_progress : bool
        If True, display tqdm progress bar
    desc : str
        Description for progress bar

    Returns:
    --------
    solution : OdeResult
        Integration result
    """
    if show_progress:
        # Create progress bar wrapper
        pbar = tqdm(total=100, desc=desc, unit='%', ncols=100,
                    bar_format='{l_bar}{bar}| {n:.1f}/{total:.0f}% [{elapsed}<{remaining}]')

        # Track last update time to avoid too many updates
        last_update = {'t': t_span[0], 'pbar_value': 0}

        def derivatives_with_progress(t, state):
            """Wrapper to update progress bar"""
            # Update progress bar
            progress = (t - t_span[0]) / (t_span[1] - t_span[0]) * 100

            # Only update if progress increased by at least 0.5%
            if progress - last_update['pbar_value'] >= 0.5:
                pbar.n = progress
                pbar.refresh()
                last_update['pbar_value'] = progress

            return orbital_derivatives(t, state, params)

        try:
            sol = solve_ivp(
                derivatives_with_progress,
                t_span,
                state0,
                method=method,
                rtol=rtol,
                atol=atol,
                dense_output=True
            )
            pbar.n = 100
            pbar.refresh()
        finally:
            pbar.close()
    else:
        # No progress bar
        sol = solve_ivp(
            orbital_derivatives,
            t_span,
            state0,
            args=(params,),
            method=method,
            rtol=rtol,
            atol=atol,
            dense_output=True
        )

    return sol



def print_error_summary(results_dict, t_eval):
    """Print error summary for all altitudes"""
    altitudes = sorted([k for k in results_dict.keys() if k != 't_eval'])
    
    for alt in altitudes:
        print(f"\n{'='*70}\nALTITUDE: {alt} km\n{'='*70}")
        
        alt_results = results_dict[alt]
        alt_t_eval = alt_results.get('t_eval', t_eval)
        
        if 'ground_truth' not in alt_results:
            continue

        r_true = alt_results['ground_truth'].sol(alt_t_eval)[:3, :].T
        v_true = alt_results['ground_truth'].sol(alt_t_eval)[3:, :].T

        for model_name in ['spherical', 'ann']:
            if model_name not in alt_results:
                continue

            r_test = alt_results[model_name].sol(alt_t_eval)[:3, :].T
            v_test = alt_results[model_name].sol(alt_t_eval)[3:, :].T

            pos_error = compute_position_error(r_true, r_test)
            vel_error = compute_velocity_error(v_true, v_test)

            print(f"\n{model_name.upper()}:")
            print(f"  Position Error (final): {pos_error[-1]/1000:.4f} km")
            print(f"  Position Error (mean):  {np.mean(pos_error)/1000:.4f} km")
            print(f"  Position Error (max):   {np.max(pos_error)/1000:.4f} km")
            print(f"  Velocity Error (final): {vel_error[-1]:.6f} m/s")
            print(f"  Velocity Error (mean):  {np.mean(vel_error):.6f} m/s")
            print(f"  Velocity Error (max):   {np.max(vel_error):.6f} m/s")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    pass