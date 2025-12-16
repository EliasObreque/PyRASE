"""
Created by Elias Obreque
Date: 16/12/2025
email: els.obrq@gmail.com
"""

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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from tqdm import tqdm
import pyvista as pv

from core.compute_perturbation_data import compute_ray_perturbation_step
from core.optimal_ray_tracing import compute_ray_tracing_fast_optimized
from core.drag_models import spherical_drag_force
from core.srp_models import spherical_srp_force

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "figure.titlesize": 18,
    "mathtext.fontset": "stix",
    "mathtext.rm": "Times New Roman",
})

# ==========================
# CONSTANTS
# ==========================

# Earth parameters
MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6378137.0  # m
J2 = 1.08262668e-3  # J2 coefficient


# ==========================
# NEURAL NETWORK MODEL
# ==========================

class MLP(nn.Module):
    """Multi-Layer Perceptron for force/torque prediction"""

    def __init__(self, in_dim, out_dim, hidden_list, activation='relu'):
        super(MLP, self).__init__()
        layers = []

        prev_dim = in_dim
        for h in hidden_list:
            layers.append(nn.Linear(prev_dim, h))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_ann_model(model_path, device='cpu'):
    """
    Load ANN model from pickle file

    Returns:
    --------
    model : torch.nn.Module
    scaler : MinMaxScaler
    """
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)

    model = model_dict['model']
    scaler = model_dict['scaler']
    model.to(device)
    model.eval()

    return model, scaler


def ann_predict(velocity_body, model, scaler, device='cpu'):
    """
    Predict force using ANN model

    Parameters:
    -----------
    velocity_body : np.ndarray (3,)
        Velocity vector in body frame [m/s]
    model : torch.nn.Module
        Trained neural network
    scaler : MinMaxScaler
        Input normalization scaler

    Returns:
    --------
    F_pred : np.ndarray (3,)
        Predicted force [N]
    """
    # Normalize input
    v_normalized = scaler.transform(velocity_body.reshape(1, -1))

    # Convert to tensor
    v_tensor = torch.tensor(v_normalized, dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        F_pred = model(v_tensor).cpu().numpy().flatten()

    return F_pred


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

    sim_data = params['sim_data']
    sim_data['alt_km'] = (r_mag - R_EARTH) / 1000
    sim_data['v_inf'] = np.linalg.norm(v_eci)

    if model_type == 'ground_truth':
        # Ray tracing model
        mesh = params['mesh']
        sim_data = params['sim_data']
        res_x = params.get('res_x', 500)
        res_y = params.get('res_y', 500)
        mass = params['mass']

        # Velocity in body frame (assume body = ECI for now, can add rotation if needed)
        v_body = v_eci
        r_inout = v_body / np.linalg.norm(v_body)

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
        sim_data = params['sim_data']
        A_ref = params['A_ref']
        mass = params['mass']

        v_body = v_eci  # Body frame = ECI frame (simplified)

        if params['include_drag']:
            F_drag = spherical_drag_force(v_body, sim_data, A_ref, mass)
            a_pert += F_drag / mass

        if params['include_srp']:
            # Sun direction (simplified: assume constant)
            sun_dir = np.array([1, 0, 0])  # Can be improved with proper ephemeris
            F_srp = spherical_srp_force(sun_dir, sim_data, A_ref)
            a_pert += F_srp / mass

    elif model_type == 'ann':
        # ANN model
        model = params['ann_model']
        scaler = params['ann_scaler']
        mass = params['mass']
        device = params.get('device', 'cpu')

        v_body = v_eci  # Body frame = ECI frame (simplified)

        # Predict force using ANN
        if params['include_drag'] or params['include_srp']:
            # ANN predicts combined force (or separate if trained separately)
            F_pred = ann_predict(v_body, model, scaler, device)
            a_pert += F_pred / mass

    # Total acceleration
    a_total = a_2body + a_j2 + a_pert

    # State derivatives
    dstate = np.concatenate([v_eci, a_total])

    return dstate


# ==========================
# ORBIT PROPAGATION
# ==========================

def propagate_orbit(state0, t_span, params, method='DOP853', rtol=1e-9, atol=1e-12):
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

    Returns:
    --------
    solution : OdeResult
        Integration result
    """
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


# ==========================
# ERROR ANALYSIS
# ==========================

def compute_position_error(r_true, r_test):
    """Compute position error magnitude"""
    return np.linalg.norm(r_true - r_test, axis=1)


def compute_velocity_error(v_true, v_test):
    """Compute velocity error magnitude"""
    return np.linalg.norm(v_true - v_test, axis=1)


def compute_relative_error(true, test):
    """Compute relative error"""
    return np.abs(true - test) / (np.abs(true) + 1e-10)


# ==========================
# VISUALIZATION
# ==========================

def plot_orbit_comparison(results_dict, t_eval, save_path='orbit_comparison.png'):
    """
    Plot orbit comparison between models

    Parameters:
    -----------
    results_dict : dict
        {
            'ground_truth': solution,
            'spherical': solution,
            'ann': solution
        }
    t_eval : np.ndarray
        Time vector [s]
    save_path : str
        Output file path
    """
    fig = plt.figure(figsize=(16, 12))

    # 3D orbit plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    colors = {
        'ground_truth': 'blue',
        'spherical': 'red',
        'ann': 'green'
    }

    labels = {
        'ground_truth': 'Ground Truth (Ray Tracing)',
        'spherical': 'Spherical Model',
        'ann': 'ANN Model'
    }

    for model_name, sol in results_dict.items():
        r = sol.sol(t_eval)[:3, :].T / 1000  # Convert to km
        ax1.plot(r[:, 0], r[:, 1], r[:, 2],
                 color=colors[model_name],
                 label=labels[model_name],
                 linewidth=2, alpha=0.8)

    # Earth sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_earth = R_EARTH / 1000 * np.outer(np.cos(u), np.sin(v))
    y_earth = R_EARTH / 1000 * np.outer(np.sin(u), np.sin(v))
    z_earth = R_EARTH / 1000 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_earth, y_earth, z_earth, color='lightblue', alpha=0.3)

    ax1.set_xlabel('X [km]')
    ax1.set_ylabel('Y [km]')
    ax1.set_zlabel('Z [km]')
    ax1.set_title('Orbit Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Position error vs time
    ax2 = fig.add_subplot(2, 2, 2)

    r_true = results_dict['ground_truth'].sol(t_eval)[:3, :].T

    for model_name in ['spherical', 'ann']:
        if model_name in results_dict:
            r_test = results_dict[model_name].sol(t_eval)[:3, :].T
            pos_error = compute_position_error(r_true, r_test)
            ax2.plot(t_eval / 3600, pos_error / 1000,
                     color=colors[model_name],
                     label=labels[model_name],
                     linewidth=2)

    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel('Position Error [km]')
    ax2.set_title('Position Error vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Velocity error vs time
    ax3 = fig.add_subplot(2, 2, 3)

    v_true = results_dict['ground_truth'].sol(t_eval)[3:, :].T

    for model_name in ['spherical', 'ann']:
        if model_name in results_dict:
            v_test = results_dict[model_name].sol(t_eval)[3:, :].T
            vel_error = compute_velocity_error(v_true, v_test)
            ax3.plot(t_eval / 3600, vel_error,
                     color=colors[model_name],
                     label=labels[model_name],
                     linewidth=2)

    ax3.set_xlabel('Time [hours]')
    ax3.set_ylabel('Velocity Error [m/s]')
    ax3.set_title('Velocity Error vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Error components
    ax4 = fig.add_subplot(2, 2, 4)

    for model_name in ['spherical', 'ann']:
        if model_name in results_dict:
            r_test = results_dict[model_name].sol(t_eval)[:3, :].T
            error_vec = r_test - r_true

            # Radial, along-track, cross-track errors
            for i, component in enumerate(['X', 'Y', 'Z']):
                ax4.plot(t_eval / 3600, error_vec[:, i] / 1000,
                         linestyle='--' if model_name == 'ann' else '-',
                         label=f'{labels[model_name]} - {component}',
                         linewidth=1.5, alpha=0.7)

    ax4.set_xlabel('Time [hours]')
    ax4.set_ylabel('Position Error Components [km]')
    ax4.set_title('Position Error Components')
    ax4.legend(fontsize=10, ncol=2)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Orbit comparison plot saved to: {save_path}")


def plot_error_statistics(results_dict, t_eval, save_path='error_statistics.png'):
    """
    Plot detailed error statistics
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    r_true = results_dict['ground_truth'].sol(t_eval)[:3, :].T
    v_true = results_dict['ground_truth'].sol(t_eval)[3:, :].T

    models = ['spherical', 'ann']
    colors = {'spherical': 'red', 'ann': 'green'}
    labels = {'spherical': 'Spherical', 'ann': 'ANN'}

    for idx, model_name in enumerate(models):
        if model_name not in results_dict:
            continue

        r_test = results_dict[model_name].sol(t_eval)[:3, :].T
        v_test = results_dict[model_name].sol(t_eval)[3:, :].T

        # Position error components
        ax = axes[idx, 0]
        error_r = r_test - r_true
        for i, comp in enumerate(['X', 'Y', 'Z']):
            ax.plot(t_eval / 3600, error_r[:, i] / 1000,
                    label=f'{comp}', linewidth=2)
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Position Error [km]')
        ax.set_title(f'{labels[model_name]} - Position Error Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Velocity error components
        ax = axes[idx, 1]
        error_v = v_test - v_true
        for i, comp in enumerate(['Vx', 'Vy', 'Vz']):
            ax.plot(t_eval / 3600, error_v[:, i],
                    label=f'{comp}', linewidth=2)
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Velocity Error [m/s]')
        ax.set_title(f'{labels[model_name]} - Velocity Error Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Total error magnitude
        ax = axes[idx, 2]
        pos_error = compute_position_error(r_true, r_test)
        vel_error = compute_velocity_error(v_true, v_test)

        ax_twin = ax.twinx()
        line1 = ax.plot(t_eval / 3600, pos_error / 1000,
                        'b-', label='Position', linewidth=2)
        line2 = ax_twin.plot(t_eval / 3600, vel_error,
                             'r-', label='Velocity', linewidth=2)

        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Position Error [km]', color='b')
        ax_twin.set_ylabel('Velocity Error [m/s]', color='r')
        ax.set_title(f'{labels[model_name]} - Total Error')
        ax.tick_params(axis='y', labelcolor='b')
        ax_twin.tick_params(axis='y', labelcolor='r')

        lines = line1 + line2
        labs = [l.get_label() for l in lines]
        ax.legend(lines, labs, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Error statistics plot saved to: {save_path}")


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
    INCLUDE_SRP = True

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
    A_ref = 0.01  # m^2 (1U CubeSat: 0.01 m^2)

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
        print(f"  Mesh loaded: {MESH_PATH}")
    else:
        print(f"  Warning: Mesh not found at {MESH_PATH}")
        print(f"  Creating simple box mesh for testing...")
        # Create simple box mesh as fallback
        mesh = pv.Cube(bounds=(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1))

    # Load ANN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ann_model, ann_scaler = load_ann_model(MODEL_PATH, device)
    print(f"  ANN model loaded: {MODEL_PATH}")
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

    sol_gt = propagate_orbit(state0, t_span, params_gt)
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

    sol_sph = propagate_orbit(state0, t_span, params_sph)
    results_dict['spherical'] = sol_sph
    print(f"   Status: {sol_sph.message}")
    print(f"   Function evaluations: {sol_sph.nfev}")

    # 3. ANN Model
    print("\n3. ANN Model...")
    params_ann = {
        'model_type': 'ann',
        'ann_model': ann_model,
        'ann_scaler': ann_scaler,
        'mass': mass,
        'include_drag': INCLUDE_DRAG,
        'include_srp': INCLUDE_SRP,
        'device': device
    }

    sol_ann = propagate_orbit(state0, t_span, params_ann)
    results_dict['ann'] = sol_ann
    print(f"   Status: {sol_ann.message}")
    print(f"   Function evaluations: {sol_ann.nfev}")

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
