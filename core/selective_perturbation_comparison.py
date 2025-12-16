"""
Created by Elias Obreque
Date: 16/12/2025
email: els.obrq@gmail.com

Provides utilities for comparing orbit propagation with selective perturbations:
- Drag only
- SRP only
- Drag + SRP combined

Generates comparison plots for each scenario.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch


# ==========================
# SELECTIVE COMPARISON
# ==========================

def compare_perturbation_scenarios(state0, t_span, t_eval, mesh, sim_data,
                                   ann_model, ann_scaler, mass, A_ref,
                                   out_dir='./results/perturbation_comparison/'):
    """
    Compare orbit propagation across different perturbation scenarios

    Scenarios:
    1. J2 only (baseline)
    2. J2 + Drag
    3. J2 + SRP
    4. J2 + Drag + SRP (full)

    For each scenario, compare:
    - Ground truth (ray tracing)
    - Spherical model
    - ANN model

    Parameters:
    -----------
    state0 : np.ndarray (6,)
        Initial state
    t_span : tuple
        (t0, tf)
    t_eval : np.ndarray
        Time evaluation points
    mesh : pv.PolyData
        Spacecraft mesh
    sim_data : dict
        Simulation parameters
    ann_model : torch.nn.Module
        Trained ANN
    ann_scaler : scaler
        Input normalization
    mass : float
        Spacecraft mass [kg]
    A_ref : float
        Reference area [m^2]
    out_dir : str
        Output directory
    """
    from orbit_propagation_comparison import propagate_orbit

    os.makedirs(out_dir, exist_ok=True)

    scenarios = [
        {'name': 'J2_only', 'drag': False, 'srp': False},
        {'name': 'J2_Drag', 'drag': True, 'srp': False},
        {'name': 'J2_SRP', 'drag': False, 'srp': True},
        {'name': 'J2_Drag_SRP', 'drag': True, 'srp': True}
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}

    print("\n" + "=" * 70)
    print("SELECTIVE PERTURBATION COMPARISON")
    print("=" * 70)

    for scenario in scenarios:
        scenario_name = scenario['name']
        include_drag = scenario['drag']
        include_srp = scenario['srp']

        print(f"\nScenario: {scenario_name}")
        print(f"  Drag: {'YES' if include_drag else 'NO'}")
        print(f"  SRP: {'YES' if include_srp else 'NO'}")

        scenario_results = {}

        # Ground Truth
        print("  - Ground Truth (Ray Tracing)...")
        params_gt = {
            'model_type': 'ground_truth',
            'mesh': mesh,
            'sim_data': sim_data,
            'A_ref': A_ref,
            'mass': mass,
            'include_drag': include_drag,
            'include_srp': include_srp,
            'res_x': 100,
            'res_y': 100
        }
        sol_gt = propagate_orbit(state0, t_span, params_gt)
        scenario_results['ground_truth'] = sol_gt

        # Spherical
        print("  - Spherical Model...")
        params_sph = {
            'model_type': 'spherical',
            'sim_data': sim_data,
            'A_ref': A_ref,
            'mass': mass,
            'include_drag': include_drag,
            'include_srp': include_srp
        }
        sol_sph = propagate_orbit(state0, t_span, params_sph)
        scenario_results['spherical'] = sol_sph

        # ANN
        print("  - ANN Model...")
        params_ann = {
            'model_type': 'ann',
            'ann_model': ann_model,
            'ann_scaler': ann_scaler,
            'mass': mass,
            'include_drag': include_drag,
            'include_srp': include_srp,
            'device': device
        }
        sol_ann = propagate_orbit(state0, t_span, params_ann)
        scenario_results['ann'] = sol_ann

        all_results[scenario_name] = scenario_results

    # Generate comparison plots
    plot_scenario_comparison(all_results, t_eval, out_dir)

    return all_results


def plot_scenario_comparison(all_results, t_eval, out_dir):
    """
    Generate comprehensive plots comparing all scenarios

    Creates:
    1. Individual scenario plots (3D orbit + errors)
    2. Cross-scenario comparison
    3. Error summary table
    """
    from orbit_propagation_comparison import (compute_position_error,
                                              compute_velocity_error)

    # Plot each scenario individually
    for scenario_name, results in all_results.items():
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig)

        # 3D orbit plot
        ax1 = fig.add_subplot(gs[0, :], projection='3d')

        colors = {
            'ground_truth': 'blue',
            'spherical': 'red',
            'ann': 'green'
        }

        labels = {
            'ground_truth': 'Ground Truth',
            'spherical': 'Spherical',
            'ann': 'ANN'
        }

        for model_name, sol in results.items():
            r = sol.sol(t_eval)[:3, :].T / 1000  # km
            ax1.plot(r[:, 0], r[:, 1], r[:, 2],
                     color=colors[model_name],
                     label=labels[model_name],
                     linewidth=2, alpha=0.8)

        # Earth
        from orbit_propagation_comparison import R_EARTH
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_earth = R_EARTH / 1000 * np.outer(np.cos(u), np.sin(v))
        y_earth = R_EARTH / 1000 * np.outer(np.sin(u), np.sin(v))
        z_earth = R_EARTH / 1000 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_surface(x_earth, y_earth, z_earth, color='lightblue', alpha=0.3)

        ax1.set_xlabel('X [km]')
        ax1.set_ylabel('Y [km]')
        ax1.set_zlabel('Z [km]')
        ax1.set_title(f'Scenario: {scenario_name.replace("_", " + ")}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Error plots
        r_true = results['ground_truth'].sol(t_eval)[:3, :].T
        v_true = results['ground_truth'].sol(t_eval)[3:, :].T

        # Position error
        ax2 = fig.add_subplot(gs[1, 0])
        for model_name in ['spherical', 'ann']:
            r_test = results[model_name].sol(t_eval)[:3, :].T
            pos_error = compute_position_error(r_true, r_test)
            ax2.plot(t_eval / 3600, pos_error / 1000,
                     color=colors[model_name],
                     label=labels[model_name],
                     linewidth=2)
        ax2.set_xlabel('Time [hours]')
        ax2.set_ylabel('Position Error [km]')
        ax2.set_title('Position Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # Velocity error
        ax3 = fig.add_subplot(gs[1, 1])
        for model_name in ['spherical', 'ann']:
            v_test = results[model_name].sol(t_eval)[3:, :].T
            vel_error = compute_velocity_error(v_true, v_test)
            ax3.plot(t_eval / 3600, vel_error,
                     color=colors[model_name],
                     label=labels[model_name],
                     linewidth=2)
        ax3.set_xlabel('Time [hours]')
        ax3.set_ylabel('Velocity Error [m/s]')
        ax3.set_title('Velocity Error')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

        # Error components
        ax4 = fig.add_subplot(gs[1, 2])
        for model_name in ['spherical', 'ann']:
            r_test = results[model_name].sol(t_eval)[:3, :].T
            error_vec = r_test - r_true
            error_mag = np.linalg.norm(error_vec, axis=1)
            ax4.plot(t_eval / 3600, error_mag / 1000,
                     color=colors[model_name],
                     label=labels[model_name],
                     linewidth=2)
        ax4.set_xlabel('Time [hours]')
        ax4.set_ylabel('Total Position Error [km]')
        ax4.set_title('Total Error Magnitude')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{scenario_name}_comparison.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Saved: {scenario_name}_comparison.png")

    # Cross-scenario comparison
    plot_cross_scenario_comparison(all_results, t_eval, out_dir)

    # Error summary
    generate_error_summary(all_results, t_eval, out_dir)


def plot_cross_scenario_comparison(all_results, t_eval, out_dir):
    """Compare how errors vary across different perturbation scenarios"""
    from orbit_propagation_comparison import (compute_position_error,
                                              compute_velocity_error)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    scenario_labels = {
        'J2_only': 'J2 Only',
        'J2_Drag': 'J2 + Drag',
        'J2_SRP': 'J2 + SRP',
        'J2_Drag_SRP': 'J2 + Drag + SRP'
    }

    colors_model = {
        'spherical': 'red',
        'ann': 'green'
    }

    markers = {
        'J2_only': 'o',
        'J2_Drag': 's',
        'J2_SRP': '^',
        'J2_Drag_SRP': 'D'
    }

    # Position error - Spherical
    ax = axes[0, 0]
    for scenario_name, results in all_results.items():
        r_true = results['ground_truth'].sol(t_eval)[:3, :].T
        r_sph = results['spherical'].sol(t_eval)[:3, :].T
        pos_error = compute_position_error(r_true, r_sph)
        ax.plot(t_eval / 3600, pos_error / 1000,
                marker=markers[scenario_name],
                markevery=50,
                label=scenario_labels[scenario_name],
                linewidth=2, alpha=0.8)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Position Error [km]')
    ax.set_title('Spherical Model - Position Error Across Scenarios')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Position error - ANN
    ax = axes[0, 1]
    for scenario_name, results in all_results.items():
        r_true = results['ground_truth'].sol(t_eval)[:3, :].T
        r_ann = results['ann'].sol(t_eval)[:3, :].T
        pos_error = compute_position_error(r_true, r_ann)
        ax.plot(t_eval / 3600, pos_error / 1000,
                marker=markers[scenario_name],
                markevery=50,
                label=scenario_labels[scenario_name],
                linewidth=2, alpha=0.8)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Position Error [km]')
    ax.set_title('ANN Model - Position Error Across Scenarios')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Velocity error - Spherical
    ax = axes[1, 0]
    for scenario_name, results in all_results.items():
        v_true = results['ground_truth'].sol(t_eval)[3:, :].T
        v_sph = results['spherical'].sol(t_eval)[3:, :].T
        vel_error = compute_velocity_error(v_true, v_sph)
        ax.plot(t_eval / 3600, vel_error,
                marker=markers[scenario_name],
                markevery=50,
                label=scenario_labels[scenario_name],
                linewidth=2, alpha=0.8)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Velocity Error [m/s]')
    ax.set_title('Spherical Model - Velocity Error Across Scenarios')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Velocity error - ANN
    ax = axes[1, 1]
    for scenario_name, results in all_results.items():
        v_true = results['ground_truth'].sol(t_eval)[3:, :].T
        v_ann = results['ann'].sol(t_eval)[3:, :].T
        vel_error = compute_velocity_error(v_true, v_ann)
        ax.plot(t_eval / 3600, vel_error,
                marker=markers[scenario_name],
                markevery=50,
                label=scenario_labels[scenario_name],
                linewidth=2, alpha=0.8)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Velocity Error [m/s]')
    ax.set_title('ANN Model - Velocity Error Across Scenarios')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cross_scenario_comparison.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: cross_scenario_comparison.png")


def generate_error_summary(all_results, t_eval, out_dir):
    """Generate numerical error summary table"""
    from orbit_propagation_comparison import (compute_position_error,
                                              compute_velocity_error)

    summary_data = []

    for scenario_name, results in all_results.items():
        r_true = results['ground_truth'].sol(t_eval)[:3, :].T
        v_true = results['ground_truth'].sol(t_eval)[3:, :].T

        for model_name in ['spherical', 'ann']:
            r_test = results[model_name].sol(t_eval)[:3, :].T
            v_test = results[model_name].sol(t_eval)[3:, :].T

            pos_error = compute_position_error(r_true, r_test)
            vel_error = compute_velocity_error(v_true, v_test)

            summary_data.append({
                'Scenario': scenario_name,
                'Model': model_name,
                'Pos_Mean_km': np.mean(pos_error) / 1000,
                'Pos_Max_km': np.max(pos_error) / 1000,
                'Pos_RMS_km': np.sqrt(np.mean(pos_error ** 2)) / 1000,
                'Vel_Mean_ms': np.mean(vel_error),
                'Vel_Max_ms': np.max(vel_error),
                'Vel_RMS_ms': np.sqrt(np.mean(vel_error ** 2))
            })

    # Save to file
    summary_file = os.path.join(out_dir, 'error_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("ERROR SUMMARY - ALL SCENARIOS\n")
        f.write("=" * 100 + "\n\n")

        for data in summary_data:
            f.write(f"Scenario: {data['Scenario']}, Model: {data['Model']}\n")
            f.write(f"  Position Error:\n")
            f.write(f"    Mean: {data['Pos_Mean_km']:.6f} km\n")
            f.write(f"    Max:  {data['Pos_Max_km']:.6f} km\n")
            f.write(f"    RMS:  {data['Pos_RMS_km']:.6f} km\n")
            f.write(f"  Velocity Error:\n")
            f.write(f"    Mean: {data['Vel_Mean_ms']:.8f} m/s\n")
            f.write(f"    Max:  {data['Vel_Max_ms']:.8f} m/s\n")
            f.write(f"    RMS:  {data['Vel_RMS_ms']:.8f} m/s\n")
            f.write("\n")

        f.write("=" * 100 + "\n")

    # Print to console
    print("\n" + "=" * 100)
    print("ERROR SUMMARY - ALL SCENARIOS")
    print("=" * 100)
    for data in summary_data:
        print(f"\n{data['Scenario']} - {data['Model'].upper()}")
        print(f"  Pos Error: Mean={data['Pos_Mean_km']:.6f} km, "
              f"Max={data['Pos_Max_km']:.6f} km, RMS={data['Pos_RMS_km']:.6f} km")
        print(f"  Vel Error: Mean={data['Vel_Mean_ms']:.8f} m/s, "
              f"Max={data['Vel_Max_ms']:.8f} m/s, RMS={data['Vel_RMS_ms']:.8f} m/s")
    print("=" * 100)

    print(f"\nError summary saved to: {summary_file}")


# ==========================
# UTILITY FUNCTIONS
# ==========================

def compute_orbital_elements(r, v):
    """
    Convert Cartesian state to orbital elements

    Returns:
    --------
    elements : dict
        {a, e, i, RAAN, omega, nu}
    """
    from orbit_propagation_comparison import MU_EARTH

    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # Specific angular momentum
    h_vec = np.cross(r, v)
    h_mag = np.linalg.norm(h_vec)

    # Node vector
    n_vec = np.cross([0, 0, 1], h_vec)
    n_mag = np.linalg.norm(n_vec)

    # Eccentricity vector
    e_vec = np.cross(v, h_vec) / MU_EARTH - r / r_mag
    e = np.linalg.norm(e_vec)

    # Specific energy
    epsilon = v_mag ** 2 / 2 - MU_EARTH / r_mag

    # Semi-major axis
    if abs(e - 1.0) > 1e-10:
        a = -MU_EARTH / (2 * epsilon)
    else:
        a = np.inf

    # Inclination
    i = np.arccos(h_vec[2] / h_mag)

    # RAAN
    if n_mag > 1e-10:
        RAAN = np.arccos(n_vec[0] / n_mag)
        if n_vec[1] < 0:
            RAAN = 2 * np.pi - RAAN
    else:
        RAAN = 0

    # Argument of periapsis
    if n_mag > 1e-10 and e > 1e-10:
        omega = np.arccos(np.dot(n_vec, e_vec) / (n_mag * e))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0

    # True anomaly
    if e > 1e-10:
        nu = np.arccos(np.dot(e_vec, r) / (e * r_mag))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0

    return {
        'a': a,
        'e': e,
        'i': i,
        'RAAN': RAAN,
        'omega': omega,
        'nu': nu
    }


def plot_orbital_element_evolution(results_dict, t_eval, save_path):
    """
    Plot evolution of orbital elements over time
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    colors = {
        'ground_truth': 'blue',
        'spherical': 'red',
        'ann': 'green'
    }

    element_names = ['a', 'e', 'i', 'RAAN', 'omega', 'nu']
    element_labels = [
        'Semi-major axis [km]',
        'Eccentricity',
        'Inclination [deg]',
        'RAAN [deg]',
        'Arg. of Perigee [deg]',
        'True Anomaly [deg]'
    ]

    for model_name, sol in results_dict.items():
        elements_over_time = {key: [] for key in element_names}

        for t in t_eval:
            state = sol.sol(t)
            r = state[:3]
            v = state[3:]

            elem = compute_orbital_elements(r, v)

            elements_over_time['a'].append(elem['a'] / 1000)  # km
            elements_over_time['e'].append(elem['e'])
            elements_over_time['i'].append(np.rad2deg(elem['i']))
            elements_over_time['RAAN'].append(np.rad2deg(elem['RAAN']))
            elements_over_time['omega'].append(np.rad2deg(elem['omega']))
            elements_over_time['nu'].append(np.rad2deg(elem['nu']))

        for idx, (elem_name, ax) in enumerate(zip(element_names, axes.flat)):
            ax.plot(t_eval / 3600, elements_over_time[elem_name],
                    color=colors[model_name],
                    label=model_name.replace('_', ' ').title(),
                    linewidth=2, alpha=0.8)
            ax.set_xlabel('Time [hours]')
            ax.set_ylabel(element_labels[idx])
            ax.set_title(f'{element_labels[idx].split("[")[0]}')
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Orbital element evolution plot saved to: {save_path}")


if __name__ == "__main__":
    print("This module is designed to be imported and used with orbit_propagation_comparison.py")
    print("\nExample usage:")
    print("  from orbit_propagation_comparison import *")
    print("  from selective_perturbation_comparison import compare_perturbation_scenarios")
    print("  ")
    print("  results = compare_perturbation_scenarios(state0, t_span, t_eval, mesh, ")
    print("                                           sim_data, ann_model, ann_scaler, ")
    print("                                           mass, A_ref)")
