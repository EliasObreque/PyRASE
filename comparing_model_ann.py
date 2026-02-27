"""
Created by Elias Obreque
Date: 16/12/2025
email: els.obrq@gmail.com
"""
from matplotlib import pyplot as plt

import os
import numpy as np
import torch
import pyvista as pv
import pickle
from datetime import datetime

# Import comparison tools
from core.orbit_propagation import (
    load_ann_model,
    propagate_orbit,
    plot_orbit_comparison,
    plot_error_statistics,
    print_error_summary,
    MU_EARTH,
    R_EARTH,
)
from core.monitor import plot_orbit_comparison, plot_error_statistics, plot_orbit_hill_frame, get_hill_frame_ON, plot_orbit, plot_altitude, plot_pos_vel_error

from core.selective_perturbation_comparison import (
    compare_perturbation_scenarios
)

print("Creating simple box mesh...")
MESH = pv.Cube(x_length=0.3, y_length=0.1, z_length=0.1)
MESH = MESH.triangulate().clean()
MESH = MESH.subdivide(1, subfilter='linear').clean()
lx, ly, lz = MESH.bounds_size
scale_matrix = np.eye(3)
scale_matrix[0, 0] = ly*lz / 2
scale_matrix[1, 1] = lx*lz / 6
scale_matrix[2, 2] = ly*lx / 3
print("Bounds:", lx, ly, lz)

# Spacecraft parameters
mass = 2934  # kg 

# ==========================
# EXAMPLE 1: BASIC COMPARISON
# ==========================

def example_basic_comparison(alt_km, n_orbits, model_name_path, config_path, mesh_path):
    """
    Basic comparison of all three models with default parameters
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: BASIC ORBIT PROPAGATION COMPARISON")
    print("=" * 70)

    # Load mesh (create simple box if file not found)
   
    mesh = MESH
    if mesh_path and os.path.exists(mesh_path):
        mesh = pv.read(mesh_path)
        print(f"\nMesh loaded: {mesh_path}")
        scale_matrix = np.eye(3)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh)
    plotter.add_axes()
    #plotter.show()
    lx, ly, lz = mesh.bounds_size
    a_ref = (lx*ly + lx*lz + ly*lz) / 3

    # Load ANN model
    optimization_path = "results/optimization/"

    model_path = optimization_path + model_name_path + config_path + "model.pkl"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ann_models = {}
    try:
        include_srp = False
        include_drag = False
        ann_model = load_ann_model(model_path, device)
        if 'drag_f' in model_path:
            include_drag = True
            ann_models['drag_f'] = ann_model
        elif 'srp_f' in model_path:
            include_srp = True
            ann_models['srp_f'] = ann_model
        print(f"ANN model loaded from: {model_path}")
    except:
        print(f"Warning: Could not load ANN model from {model_path}")
        print("Will only compare ground truth and spherical models")
        ann_model = None
        ann_scaler = None

    # Define initial orbit (alt_km km circular LEO, 51.6° inclination)
    # alt_km = 450

    # Output directory
    out_dir = "./results/example_basic/" + model_name_path + config_path
    os.makedirs(out_dir, exist_ok=True)
    

    a = R_EARTH + alt_km * 1e3  # m
    e = 0.001
    i = np.deg2rad(80.0)

    # Simple initial state (circular orbit in equatorial plane)
    r0 = np.array([a, 0, 0])
    v0 = np.array([0, np.sqrt(MU_EARTH / a), 0])

    # Apply inclination rotation
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    R_i = np.array([
        [1, 0, 0],
        [0, cos_i, -sin_i],
        [0, sin_i, cos_i]
    ])

    r0 = R_i @ r0
    v0 = R_i @ v0

    rho_ = get_hill_frame_ON(r0, v0) @ np.array([1, 1, 1.0])

    state0 = np.concatenate([r0, v0])

    # Propagation time (1 orbit)
    orbital_period = 2 * np.pi * np.sqrt(a ** 3 / MU_EARTH)
    t_span = (0, n_orbits * orbital_period)
    t_eval = np.linspace(0, t_span[1], int(100 * n_orbits))

    print(f"\nInitial orbit:")
    print(f"  Altitude: {(a - R_EARTH) / 1000:.2f} km")
    print(f"  Period: {orbital_period / 60:.2f} minutes")
    print(f"  Inclination: {np.rad2deg(i):.1f} degrees")
    

    t_ = '2025-01-01 12:00:00'
    dt = datetime.strptime(t_, '%Y-%m-%d %H:%M:%S')

    # Convert to Julian Date
    jd_ = 2440587.5 + dt.timestamp() / 86400.0

    # Simulation parameters
    sim_data = {
        'v_inf': np.linalg.norm(v0),
        'alt_km': (a - R_EARTH) / 1000,
        'time_str': t_,
        'jd': jd_,
        'sigma_N': 0.8,
        'sigma_T': 0.5,
        'T_wall': 300,
        'A_ref': a_ref,
        'spec_srp': 0.2,
        'diffuse_srp': 0.6,
        'scale_shape': scale_matrix,
    }

    # Propagate orbits
    results = {}
    # 1. Nominal
    print("\nPropagating Nominal...")
    params_gt = {
        'model_type': 'nominal_j2',
        'mesh': None,
        'sim_data': sim_data,
        'A_ref': a_ref,
        'ann_models': None,
        'mass': mass,
        'include_drag': include_drag,
        'include_srp': include_srp,
        'res_x': 500,  # Low resolution for speed
        'res_y': 500
    }
    results['nominal_j2'] = propagate_orbit(state0, t_span, params_gt)

    # 2. Spherical Model
    print("\nPropagating Spherical Model...")
    params_sph = {
        'model_type': 'spherical',
        'sim_data': sim_data,
        'A_ref': a_ref,
        'mass': mass,
        'include_drag': include_drag,
        'include_srp': include_srp
    }
    results['spherical'] = propagate_orbit(state0, t_span, params_sph)

    # 3. ANN Model (if available)
    if ann_models is not None:
        print("\nPropagating ANN Model...")
        params_ann = {
            'model_type': 'ann',
            'ann_models': ann_models,
            'sim_data': sim_data,
            'mass': mass,
            'include_drag': include_drag,
            'include_srp': include_srp,
            'device': device
        }
        results['ann'] = propagate_orbit(state0, t_span, params_ann)

    # 1. Ground Truth
    print("\nPropagating Ground Truth (Ray Tracing)...")
    params_gt = {
        'model_type': 'ground_truth',
        'mesh': mesh,
        'sim_data': sim_data,
        'A_ref': a_ref,
        'ann_models': ann_models,
        'mass': mass,
        'include_drag': include_drag,
        'include_srp': include_srp,
        'res_x': 1000,  # Low resolution for speed
        'res_y': 1000
    }
    results['ground_truth'] = propagate_orbit(state0, t_span, params_gt)

    print("\nPropagating Ground Truth (Theory)...")
    params_gt = {
        'model_type': 'ground_truth_theory',
        'mesh': mesh,
        'sim_data': sim_data,
        'A_ref': a_ref,
        'ann_models': None,
        'mass': mass,
        'include_drag': include_drag,
        'include_srp': include_srp,
        'lx': lx,
        'ly': ly,
        'lz': lz,
    }
    #results['ground_truth'] = propagate_orbit(state0, t_span, params_gt)
    results['t_eval'] = t_eval
    return results

# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ORBIT PROPAGATION COMPARISON - EXAMPLE USAGE")
    print("=" * 70)
       
    # -----------------------------------------------------
    # user option
    n_orbits = 16


    out_dir = "./results/example_basic/"
    os.makedirs(out_dir, exist_ok=True)
    name = f"rect_prism_{n_orbits}.pkl"
    
    #model_name_path = "/aqua_b_data_1000_sample_20000/drag_f/"
    model_name_path = "/rect_prism_data_1000_sample_10000/drag_f/"
    
    config_path = "/config_239/" # aqua t
    #config_path = "/config_168/" # rec

    mesh_path = "./models/Aqua+(B).stl"
    #mesh_path = None
    # Output directory
    out_dir = "./results/example_basic/" + model_name_path + config_path
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(out_dir + name):
        results_300 = example_basic_comparison(500.0, n_orbits, model_name_path, config_path, mesh_path)
        results_400 = example_basic_comparison(1000.0, n_orbits, model_name_path, config_path, mesh_path)
        results_500 = example_basic_comparison(1500.0, n_orbits, model_name_path, config_path, mesh_path)
        t_eval = results_500['t_eval']
        results = {'500': results_300,
                   '1000': results_400,
                   '1500': results_500,
                   't_eval': t_eval
                   }
        t_eval = results['t_eval']
        with open(out_dir + name, 'wb') as f:
            pickle.dump(results, f)
    else:
        with open(out_dir + name, 'rb') as f:
            results = pickle.load(f)
            t_eval = results['t_eval']

    # Generate plots
    print("\nGenerating comparison plots...")
    #plot_orbit_comparison(results, t_eval, save_path=os.path.join(out_dir, 'orbit_comparison.png'))
    plot_pos_vel_error(results, t_eval, save_path=os.path.join(out_dir, 'pos_vel_error.png'), log_scale=True)
    #plot_orbit_hill_frame(results, t_eval, os.path.join(out_dir, 'hill_orbit_comparison.png'))

    #if len(results) > 2:  # If ANN model is available
    #    plot_error_statistics(results, t_eval,
    #                            save_path=os.path.join(out_dir, 'errors.png'))

    # Print error summary
    print_error_summary(results, t_eval)
    plt.show()

    print("\n" + "=" * 70)
    print("✓ EXAMPLES COMPLETED")
    print("=" * 70)
