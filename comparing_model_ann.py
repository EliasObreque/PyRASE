"""
Created by Elias Obreque
Date: 16/12/2025
email: els.obrq@gmail.com
"""

# -*- coding: utf-8 -*-
"""
Example Usage Script for Orbit Propagation Comparison
Created by: O Break
Date: 2025-12-16

Demonstrates:
1. Basic orbit propagation comparison
2. Selective perturbation analysis
3. Custom configuration
4. Results visualization
"""

import os
import numpy as np
import torch
import pyvista as pv
import pickle

# Import comparison tools
from core.orbit_propagation import (
    load_ann_model,
    propagate_orbit,
    plot_orbit_comparison,
    plot_error_statistics,
    print_error_summary,
    MU_EARTH,
    R_EARTH
)

from core.selective_perturbation_comparison import (
    compare_perturbation_scenarios
)

print("Creating simple box mesh...")
MESH = pv.Cube(x_length=3, y_length=1, z_length=2)
MESH = MESH.triangulate().clean()
MESH = MESH.subdivide(1, subfilter='linear').clean()

# Spacecraft parameters
mass = 10.0  # kg (3U CubeSat)
A_ref = (3 + 2 + 1) / 3  # m^2 (3U CubeSat: 0.03 m^2)

# ==========================
# EXAMPLE 1: BASIC COMPARISON
# ==========================

def example_basic_comparison():
    """
    Basic comparison of all three models with default parameters
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: BASIC ORBIT PROPAGATION COMPARISON")
    print("=" * 70)

    # Output directory
    out_dir = "./results/example_basic/"
    os.makedirs(out_dir, exist_ok=True)

    # Load mesh (create simple box if file not found)
    mesh_path = "./mesh/spacecraft.stl"
    mesh = MESH
    if os.path.exists(mesh_path):
        mesh = pv.read(mesh_path)
        print(f"Mesh loaded: {mesh_path}")

    # Load ANN model
    model_path = "results/optimization/rect_prism_data_1000_sample_10000/config_21/model_drag_f.pkl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ann_models = {}
    try:
        ann_model = load_ann_model(model_path, device)
        ann_models['drag_f'] = ann_model
        print(f"ANN model loaded from: {model_path}")
    except:
        print(f"Warning: Could not load ANN model from {model_path}")
        print("Will only compare ground truth and spherical models")
        ann_model = None
        ann_scaler = None

    # Define initial orbit (400 km circular LEO, 51.6° inclination)
    a = R_EARTH + 400e3  # m
    e = 0.001
    i = np.deg2rad(51.6)

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

    state0 = np.concatenate([r0, v0])

    # Propagation time (1 orbit)
    orbital_period = 2 * np.pi * np.sqrt(a ** 3 / MU_EARTH)
    t_span = (0, 15 * orbital_period)
    t_eval = np.linspace(0, t_span[1], 200)

    print(f"\nInitial orbit:")
    print(f"  Altitude: {(a - R_EARTH) / 1000:.2f} km")
    print(f"  Period: {orbital_period / 60:.2f} minutes")
    print(f"  Inclination: {np.rad2deg(i):.1f} degrees")

    # Simulation parameters
    sim_data = {
        'v_inf': np.linalg.norm(v0),
        'alt_km': (a - R_EARTH) / 1000,
        'time_str': '2024-01-01 12:00:00',
        'sigma_N': 0.9,
        'sigma_T': 0.9,
        'T_wall': 300,
        'A_ref': A_ref,
        'spec_srp': 0.1,
        'diffuse_srp': 0.5
    }

    # Propagate orbits
    results = {}

    # 1. Ground Truth
    print("\nPropagating Ground Truth (Ray Tracing)...")
    params_gt = {
        'model_type': 'ground_truth',
        'mesh': mesh,
        'sim_data': sim_data,
        'A_ref': A_ref,
        'mass': mass,
        'include_drag': True,
        'include_srp': True,
        'res_x': 100,  # Low resolution for speed
        'res_y': 100
    }
    results['ground_truth'] = propagate_orbit(state0, t_span, params_gt)

    # 2. Spherical Model
    print("Propagating Spherical Model...")
    params_sph = {
        'model_type': 'spherical',
        'sim_data': sim_data,
        'A_ref': A_ref,
        'mass': mass,
        'include_drag': True,
        'include_srp': True
    }
    results['spherical'] = propagate_orbit(state0, t_span, params_sph)

    # 3. ANN Model (if available)
    if ann_models is not None:
        print("Propagating ANN Model...")
        params_ann = {
            'model_type': 'ann',
            'ann_models': ann_models,
            'sim_data': sim_data,
            'mass': mass,
            'include_drag': True,
            'include_srp': True,
            'device': device
        }
        results['ann'] = propagate_orbit(state0, t_span, params_ann)

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_orbit_comparison(results, t_eval,
                          save_path=os.path.join(out_dir, 'comparison.png'))

    if len(results) > 2:  # If ANN model is available
        plot_error_statistics(results, t_eval,
                              save_path=os.path.join(out_dir, 'errors.png'))

    # Print error summary
    print_error_summary(results, t_eval)

    print(f"\nResults saved to: {out_dir}")
    print("=" * 70)

    return results, t_eval



# ==========================
# EXAMPLE 3: CUSTOM CONFIGURATION
# ==========================

def example_custom_config():
    """
    Demonstrate custom configuration options
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: CUSTOM CONFIGURATION")
    print("=" * 70)

    out_dir = "./results/example_custom/"
    os.makedirs(out_dir, exist_ok=True)

    # Custom orbit: Highly elliptical (Molniya-like)
    a = R_EARTH + 20000e3  # Semi-major axis
    e = 0.7  # High eccentricity
    i = np.deg2rad(63.4)  # Critical inclination

    # Convert to Cartesian (at perigee)
    r_perigee = a * (1 - e)
    v_perigee = np.sqrt(MU_EARTH * (2 / r_perigee - 1 / a))

    r0 = np.array([r_perigee, 0, 0])
    v0 = np.array([0, v_perigee, 0])

    # Apply inclination
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    R_i = np.array([
        [1, 0, 0],
        [0, cos_i, -sin_i],
        [0, sin_i, cos_i]
    ])

    r0 = R_i @ r0
    v0 = R_i @ v0
    state0 = np.concatenate([r0, v0])

    # Propagation (half orbit)
    orbital_period = 2 * np.pi * np.sqrt(a ** 3 / MU_EARTH)
    t_span = (0, 0.5 * orbital_period)
    t_eval = np.linspace(0, t_span[1], 500)

    print(f"\nCustom orbit:")
    print(f"  Semi-major axis: {a / 1000:.0f} km")
    print(f"  Eccentricity: {e:.2f}")
    print(f"  Perigee altitude: {(r_perigee - R_EARTH) / 1000:.0f} km")
    print(f"  Apogee altitude: {(a * (1 + e) - R_EARTH) / 1000:.0f} km")
    print(f"  Period: {orbital_period / 3600:.2f} hours")


    # Custom atmospheric parameters (higher altitude)
    sim_data = {
        'v_inf': v_perigee,
        'alt_km': (r_perigee - R_EARTH) / 1000,
        'time_str': '2024-06-15 12:00:00',  # Different date
        'sigma_N': 0.95,  # Higher accommodation
        'sigma_T': 0.95,
        'T_wall': 320,
        'A_ref': A_ref,
        'spec_srp': 0.2,  # More specular reflection
        'diffuse_srp': 0.4
    }

    # Model
    try:
        model_path = "./models/model.pkl"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ann_model, ann_scaler = load_ann_model(model_path, device)
    except:
        print("ANN model not available")
        ann_model = None
        ann_scaler = None

    # Propagate with custom integration settings
    print("\nPropagating with high-precision settings...")

    results = {}

    # Spherical model with custom tolerances
    params_sph = {
        'model_type': 'spherical',
        'sim_data': sim_data,
        'A_ref': A_ref,
        'mass': mass,
        'include_drag': True,
        'include_srp': True
    }

    results['spherical'] = propagate_orbit(
        state0, t_span, params_sph,
        method='DOP853',
        rtol=1e-12,  # Very tight tolerance
        atol=1e-15
    )

    print("Custom configuration completed!")
    print(f"Results saved to: {out_dir}")
    print("=" * 70)

    return results


# ==========================
# EXAMPLE 4: DRAG ONLY VS SRP ONLY
# ==========================

def example_drag_vs_srp():
    """
    Compare relative importance of drag vs SRP
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: DRAG VS SRP COMPARISON")
    print("=" * 70)

    out_dir = "./results/example_drag_vs_srp/"
    os.makedirs(out_dir, exist_ok=True)

    # Setup
    a = R_EARTH + 400e3
    state0 = np.array([a, 0, 0, 0, np.sqrt(MU_EARTH / a), 0])

    orbital_period = 2 * np.pi * np.sqrt(a ** 3 / MU_EARTH)
    t_span = (0, 10 * orbital_period)  # 10 orbits
    t_eval = np.linspace(0, t_span[1], 1000)

    mass = 10.0
    A_ref = 0.03

    sim_data = {
        'v_inf': np.sqrt(MU_EARTH / a),
        'alt_km': 400,
        'time_str': '2024-01-01 12:00:00',
        'sigma_N': 0.9,
        'sigma_T': 0.9,
        'T_wall': 300,
        'A_ref': A_ref,
        'spec_srp': 0.1,
        'diffuse_srp': 0.5
    }

    print("\nPropagating 4 scenarios (spherical model):")
    print("  1. J2 only (baseline)")
    print("  2. J2 + Drag")
    print("  3. J2 + SRP")
    print("  4. J2 + Drag + SRP")

    scenarios = {
        'J2_only': {'drag': False, 'srp': False},
        'J2_Drag': {'drag': True, 'srp': False},
        'J2_SRP': {'drag': False, 'srp': True},
        'J2_Drag_SRP': {'drag': True, 'srp': True}
    }

    results = {}

    for name, config in scenarios.items():
        print(f"\n  Propagating: {name}")
        params = {
            'model_type': 'spherical',
            'sim_data': sim_data,
            'A_ref': A_ref,
            'mass': mass,
            'include_drag': config['drag'],
            'include_srp': config['srp']
        }
        results[name] = propagate_orbit(state0, t_span, params)

    # Calculate altitude decay
    print("\n" + "=" * 70)
    print("ALTITUDE DECAY ANALYSIS")
    print("=" * 70)

    for name, sol in results.items():
        r_final = sol.sol(t_span[1])[:3]
        alt_initial = (np.linalg.norm(state0[:3]) - R_EARTH) / 1000
        alt_final = (np.linalg.norm(r_final) - R_EARTH) / 1000
        decay = alt_initial - alt_final

        print(f"\n{name}:")
        print(f"  Initial altitude: {alt_initial:.3f} km")
        print(f"  Final altitude:   {alt_final:.3f} km")
        print(f"  Decay:            {decay:.3f} km")
        print(f"  Decay rate:       {decay / (t_span[1] / 86400):.3f} km/day")

    print("\n" + "=" * 70)

    return results


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ORBIT PROPAGATION COMPARISON - EXAMPLE USAGE")
    print("=" * 70)

    # Run examples
    print("\nSelect example to run:")
    print("  1. Basic comparison (default)")
    print("  2. Custom configuration")
    print("  3. Drag vs SRP comparison")
    print("  4. Run all examples")

    choice = input("\nEnter choice (1-5) or press Enter for example 1: ").strip()

    if choice == '' or choice == '1':
        example_basic_comparison()

    elif choice == '2':
        example_custom_config()

    elif choice == '3':
        example_drag_vs_srp()

    elif choice == '4':
        print("\nRunning all examples...")
        example_basic_comparison()
        # example_selective_perturbations()  # Commented out - very time consuming
        example_custom_config()
        example_drag_vs_srp()

    else:
        print("Invalid choice. Running example 1...")
        example_basic_comparison()

    print("\n" + "=" * 70)
    print("✓ EXAMPLES COMPLETED")
    print("=" * 70)
