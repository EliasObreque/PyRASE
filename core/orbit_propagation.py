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
import numpy as np
import torch
from scipy.integrate import solve_ivp
from tqdm import tqdm
import pyvista as pv

from core.compute_perturbation_data import compute_ray_perturbation_step
from core.optimal_ray_tracing import compute_ray_tracing_fast_optimized
from core.drag_models import spherical_drag_force, get_dynamic_pressure
from core.srp_models import spherical_srp_force
from core.ann_tools import (load_ann_models, load_ann_model,
                            ann_predict_force, ann_predict_torque, ann_predict)
from core.monitor import plot_orbit_comparison, plot_error_statistics, compute_position_error, compute_velocity_error

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
    a_j2 = j2_acceleration(r_eci)

    # Perturbation acceleration
    a_pert = np.zeros(3)

    model_type = params['model_type']

    # CRITICAL: Update altitude for atmospheric density calculations
    # This must be done before any drag force computation
    # TODO: attitude

    sim_data = params['sim_data']
    sim_data['alt_km'] = (r_mag - R_EARTH) / 1000
    sim_data['v_inf'] = np.linalg.norm(v_eci)

    if model_type == 'ground_truth':
        # Ray tracing model
        mesh = params['mesh']
        res_x = params.get('res_x', 500)
        res_y = params.get('res_y', 500)
        mass = params['mass']

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

            if params['include_drag']:
                a_pert += F_drag_total / mass
            if params['include_srp']:
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
            # TODO: Can be improved with proper ephemeris
            sun_dir = np.array([1, 0, 0])
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
            F_drag = q_ * ann_predict_force(v_body, ann_models.get('drag_f'), device)
            T_drag = q_ * ann_predict_torque(v_body, ann_models.get('drag_t'), device)

            a_pert += F_drag / mass
            # Note: Torque doesn't affect translational motion directly

        # Predict SRP force and torque
        if params['include_srp']:
            # For SRP, we need sun direction (simplified here)
            sun_dir = np.array([1, 0, 0])

            P_SRP = 4.56e-6  # N/m^2
            F_srp = P_SRP * ann_predict_force(sun_dir, ann_models.get('srp_f'), device)
            T_srp = P_SRP * ann_predict_torque(sun_dir, ann_models.get('srp_t'), device)
            a_pert += F_srp / mass
            # Note: Torque doesn't affect translational motion directly

    # Total acceleration
    a_total = a_2body + a_j2 + a_pert

    # State derivatives
    dstate = np.concatenate([v_eci, a_total])

    return dstate


# ==========================
# ORBIT PROPAGATION
# ==========================

def propagate_orbit(state0, t_span, params, method='DOP853', rtol=1e-9, atol=1e-12,
                    show_progress=True, desc="Propagating orbit"):
    """
    Propagate orbit with specified perturbations

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
    """Print numerical error summary"""
    print("\n" + "=" * 70)
    print("ERROR SUMMARY")
    print("=" * 70)

    r_true = results_dict['ground_truth'].sol(t_eval)[:3, :].T
    v_true = results_dict['ground_truth'].sol(t_eval)[3:, :].T

    for model_name in ['spherical', 'ann']:
        if model_name not in results_dict:
            continue

        r_test = results_dict[model_name].sol(t_eval)[:3, :].T
        v_test = results_dict[model_name].sol(t_eval)[3:, :].T

        pos_error = compute_position_error(r_true, r_test)
        vel_error = compute_velocity_error(v_true, v_test)

        print(f"\n{model_name.upper()} MODEL:")
        print(f"  Position Error:")
        print(f"    Mean: {np.mean(pos_error) / 1000:.3f} km")
        print(f"    Max:  {np.max(pos_error) / 1000:.3f} km")
        print(f"    RMS:  {np.sqrt(np.mean(pos_error ** 2)) / 1000:.3f} km")
        print(f"  Velocity Error:")
        print(f"    Mean: {np.mean(vel_error):.6f} m/s")
        print(f"    Max:  {np.max(vel_error):.6f} m/s")
        print(f"    RMS:  {np.sqrt(np.mean(vel_error ** 2)):.6f} m/s")

    print("=" * 70)


# ==========================
# MAIN EXECUTION
# ==========================

def main():
    """Main execution function"""

    print("\n" + "=" * 70)
    print("ORBIT PROPAGATION COMPARISON")
    print("=" * 70)

    # ==========================
    # CONFIGURATION
    # ==========================

    # Output directory
    OUT_DIR = "./results/orbit_comparison/"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Mesh file (for ground truth)
    MESH_PATH = "./mesh/spacecraft.stl"  # Update with actual path

    # ANN model path
    MODEL_PATH = "/mnt/user-data/uploads/model.pkl"

    # Simulation parameters
    INCLUDE_DRAG = True
    INCLUDE_SRP = False

    # Initial orbital elements (example: LEO)
    a = R_EARTH + 400e3  # Semi-major axis [m]
    e = 0.001  # Eccentricity
    i = np.deg2rad(51.6)  # Inclination [rad]
    RAAN = np.deg2rad(0)  # Right ascension of ascending node [rad]
    omega = np.deg2rad(0)  # Argument of perigee [rad]
    nu = np.deg2rad(0)  # True anomaly [rad]

    # Convert to Cartesian state
    p = a * (1 - e ** 2)
    r_mag = p / (1 + e * np.cos(nu))

    # Perifocal frame
    r_pf = r_mag * np.array([np.cos(nu), np.sin(nu), 0])
    v_pf = np.sqrt(MU_EARTH / p) * np.array([-np.sin(nu), e + np.cos(nu), 0])

    # Rotation matrices
    R3_RAAN = np.array([
        [np.cos(RAAN), -np.sin(RAAN), 0],
        [np.sin(RAAN), np.cos(RAAN), 0],
        [0, 0, 1]
    ])

    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])

    R3_omega = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])

    # Transform to ECI
    Q = R3_RAAN @ R1_i @ R3_omega
    r0 = Q @ r_pf
    v0 = Q @ v_pf

    state0 = np.concatenate([r0, v0])

    print(f"\nInitial State:")
    print(f"  Position: {r0 / 1000} km")
    print(f"  Velocity: {v0} m/s")
    print(f"  Altitude: {(np.linalg.norm(r0) - R_EARTH) / 1000:.2f} km")

    # Propagation time
    n_orbits = 2
    orbital_period = 2 * np.pi * np.sqrt(a ** 3 / MU_EARTH)
    t_final = n_orbits * orbital_period
    t_span = (0, t_final)
    t_eval = np.linspace(0, t_final, 500)

    print(f"\nPropagation:")
    print(f"  Duration: {n_orbits} orbits ({t_final / 3600:.2f} hours)")
    print(f"  Orbital period: {orbital_period / 60:.2f} minutes")

    # Spacecraft parameters
    mass = 10.0  # kg
    A_ref = (3 + 2 + 1) / 2

    # Simulation data for atmospheric/SRP models
    sim_data = {
        'v_inf': 7700,  # m/s (approximate LEO velocity)
        'alt_km': 400,
        'time_str': '2024-01-01 12:00:00',
        'sigma_N': 0.9,
        'sigma_T': 0.9,
        'T_wall': 300,  # K
        'A_ref': A_ref,
        'spec_srp': 0.1,
        'diffuse_srp': 0.5
    }

    print(f"\nSpacecraft:")
    print(f"  Mass: {mass} kg")
    print(f"  Reference area: {A_ref} m^2")
    print(f"  Altitude: {sim_data['alt_km']} km")

    print(f"\nPerturbations:")
    print(f"  Drag: {'YES' if INCLUDE_DRAG else 'NO'}")
    print(f"  SRP: {'YES' if INCLUDE_SRP else 'NO'}")
    print(f"  J2: YES (always included)")

    # ==========================
    # LOAD MODELS
    # ==========================

    print("\nLoading models...")

    # Load mesh for ground truth
    mesh = None
    if os.path.exists(MESH_PATH):
        mesh = pv.read(MESH_PATH)
        mesh = mesh.triangulate().clean()
        mesh = mesh.subdivide(1, subfilter='linear').clean()
        print(f"  Mesh loaded: {MESH_PATH}")
    else:
        print(f"  Warning: Mesh not found at {MESH_PATH}")
        print(f"  Creating rectangular prism mesh for testing...")
        # Create rectangular prism mesh as fallback
        mesh = pv.Cube(x_length=3, y_length=1, z_length=2)
        mesh = mesh.triangulate().clean()
        mesh = mesh.subdivide(1, subfilter='linear').clean()

    # Load ANN models (4 separate models: drag_f, drag_t, srp_f, srp_t)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model paths
    MODEL_DIR_DRAG_F = "./results/optimization/rect_prism_data_1000_sample_10000/config_21/"
    MODEL_DIR_DRAG_T = "./results/optimization/rect_prism_data_1000_sample_10000/config_22/"
    MODEL_DIR_SRP_F = "./results/optimization/rect_prism_data_1000_sample_10000/config_23/"
    MODEL_DIR_SRP_T = "./results/optimization/rect_prism_data_1000_sample_10000/config_24/"

    model_paths = {
        'drag_f': os.path.join(MODEL_DIR_DRAG_F, 'model_drag_f.pkl'),
        'drag_t': os.path.join(MODEL_DIR_DRAG_T, 'model_drag_t.pkl'),
        'srp_f': os.path.join(MODEL_DIR_SRP_F, 'model_srp_f.pkl'),
        'srp_t': os.path.join(MODEL_DIR_SRP_T, 'model_srp_t.pkl')
    }

    ann_models = load_ann_models(model_paths, device)

    if ann_models:
        print(f"  Device: {device}")

    # ==========================
    # PROPAGATE ORBITS
    # ==========================

    results_dict = {}

    print("\n" + "=" * 70)
    print("PROPAGATING ORBITS")
    print("=" * 70)

    # 1. Ground Truth (Ray Tracing)
    print("\n1. Ground Truth (Ray Tracing)...")
    params_gt = {
        'model_type': 'ground_truth',
        'mesh': mesh,
        'sim_data': sim_data,
        'A_ref': A_ref,
        'mass': mass,
        'include_drag': INCLUDE_DRAG,
        'include_srp': INCLUDE_SRP,
        'res_x': 100,  # Reduced resolution for faster computation
        'res_y': 100
    }

    sol_gt = propagate_orbit(state0, t_span, params_gt,
                             show_progress=True,
                             desc="Ground Truth (Ray Tracing)")
    results_dict['ground_truth'] = sol_gt
    print(f"   Status: {sol_gt.message}")
    print(f"   Function evaluations: {sol_gt.nfev}")

    # 2. Spherical Model
    print("\n2. Spherical Model...")
    params_sph = {
        'model_type': 'spherical',
        'sim_data': sim_data,
        'A_ref': A_ref,
        'mass': mass,
        'include_drag': INCLUDE_DRAG,
        'include_srp': INCLUDE_SRP
    }

    sol_sph = propagate_orbit(state0, t_span, params_sph,
                              show_progress=True,
                              desc="Spherical Model")
    results_dict['spherical'] = sol_sph
    print(f"   Status: {sol_sph.message}")
    print(f"   Function evaluations: {sol_sph.nfev}")

    # 3. ANN Model
    if ann_models is not None:
        print("\n3. ANN Model...")
        params_ann = {
            'model_type': 'ann',
            'ann_models': ann_models,
            'mass': mass,
            'include_drag': INCLUDE_DRAG,
            'include_srp': INCLUDE_SRP,
            'device': device
        }

        sol_ann = propagate_orbit(state0, t_span, params_ann,
                                  show_progress=True,
                                  desc="ANN Model")
        results_dict['ann'] = sol_ann
        print(f"   Status: {sol_ann.message}")
        print(f"   Function evaluations: {sol_ann.nfev}")
    else:
        print("\n3. ANN Model: SKIPPED (models not loaded)")

    # ==========================
    # ERROR ANALYSIS
    # ==========================

    print_error_summary(results_dict, t_eval)

    # ==========================
    # VISUALIZATION
    # ==========================

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_orbit_comparison(results_dict, t_eval,
                          save_path=os.path.join(OUT_DIR, 'orbit_comparison.png'))

    plot_error_statistics(results_dict, t_eval,
                          save_path=os.path.join(OUT_DIR, 'error_statistics.png'))

    # Save results
    results_file = os.path.join(OUT_DIR, 'propagation_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump({
            'results_dict': results_dict,
            't_eval': t_eval,
            'state0': state0,
            'params': {
                'mass': mass,
                'A_ref': A_ref,
                'sim_data': sim_data,
                'include_drag': INCLUDE_DRAG,
                'include_srp': INCLUDE_SRP
            }
        }, f)
    print(f"\nResults saved to: {results_file}")

    print("\n" + "=" * 70)
    print("✓ ORBIT PROPAGATION COMPARISON COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()