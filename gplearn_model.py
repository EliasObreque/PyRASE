# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 15:04:19 2025

@author: mndc5
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils.validation import check_X_y, check_array
from gplearn.fitness import make_fitness

DATA_PATH = "./results/data/rect_prism_data_1000_sample_10000"
OUT_DIR = "../gplearn_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

COL_OUT_DRAG_F = ['Fx_drag', 'Fy_drag', 'Fz_drag']
COL_OUT_DRAG_T = ['Tx_drag', 'Ty_drag', 'Tz_drag']


def _validate_data(self, X, y=None, **check_params):
    if y is None:
        X = check_array(X, **check_params)
        self.n_features_in_ = X.shape[1]
        return X
    X, y = check_X_y(X, y, **check_params)
    self.n_features_in_ = X.shape[1]
    return X, y


SymbolicRegressor._validate_data = _validate_data


def load_data(data_path, outputs=None):
    """Load and prepare data"""
    with open(data_path, "rb") as f:
        data_mesh = pickle.load(f)

    r_array = np.array(data_mesh['r_'])

    X = r_array

    if outputs is None:
        outputs = COL_OUT_DRAG_F + COL_OUT_DRAG_T

    y_dict = {
        'Fx_drag': np.array(data_mesh['F_drag'])[:, 0],
        'Fy_drag': np.array(data_mesh['F_drag'])[:, 1],
        'Fz_drag': np.array(data_mesh['F_drag'])[:, 2],
        'Tx_drag': np.array(data_mesh['T_drag'])[:, 0],
        'Ty_drag': np.array(data_mesh['T_drag'])[:, 1],
        'Tz_drag': np.array(data_mesh['T_drag'])[:, 2],
    }

    y = np.column_stack([y_dict[out] for out in outputs])

    return X, y, outputs


def engineer_features(X):
    """Minimal features for unit vector input"""
    v_x, v_y, v_z = X[:, 0], X[:, 1], X[:, 2]

    return np.column_stack([
        v_x, v_y, v_z,  # Raw components
        v_y * v_z,  # Cross product for Tx
        v_z * v_x,  # Cross product for Ty
        v_x * v_y,  # Cross product for Tz

    ])


def _r2_fitness(y, y_pred, w):
    """Custom fitness based on R² (higher is better, so we return negative)"""
    return -r2_score(y, y_pred)  # Negative because gplearn minimizes


r2_fitness = make_fitness(function=_r2_fitness, greater_is_better=False)


def create_regressor(config=None):
    """Create SymbolicRegressor with configuration"""
    default_config = {
        'population_size': 5000,
        'generations': 1000,
        'tournament_size': 20,
        'stopping_criteria': 0.00001,
        'const_range': (-1., 1.),
        'init_depth': (2, 6),
        'init_method': 'half and half',
        'function_set': ['add', 'sub', 'mul', 'div', 'sqrt', 'abs'],
        'metric': r2_fitness,
        'parsimony_coefficient': 0.001,
        'p_crossover': 0.7,
        'p_subtree_mutation': 0.1,
        'p_hoist_mutation': 0.05,
        'p_point_mutation': 0.1,
        'max_samples': 0.9,
        'verbose': 1,
        'n_jobs': 6,
        'random_state': 42
    }

    if config:
        default_config.update(config)

    return SymbolicRegressor(**default_config)


def fit_models(X_train, y_train, output_names, config=None):
    """Fit symbolic regression models for each output"""
    X_eng = engineer_features(X_train)
    models = {}

    for i, name in enumerate(output_names):
        print(f"\nFitting {name}...")
        model = create_regressor(config)
        model.fit(X_eng, y_train[:, i])
        models[name] = model

        print(f"  Program: {model._program}")
        print(f"  Fitness: {model._program.fitness_:.6f}")

    return models


def predict_all(models, X, output_names):
    """Predict all outputs"""
    X_eng = engineer_features(X)
    return np.column_stack([models[name].predict(X_eng) for name in output_names])


def evaluate(models, X_test, y_test, output_names):
    """Evaluate models"""
    y_pred = predict_all(models, X_test, output_names)

    print("\nTest Set Performance:")
    results = {}
    for i, name in enumerate(output_names):
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        results[name] = {'r2': r2, 'rmse': rmse}
        print(f"{name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")

    return results, y_pred


def plot_evolution(models, output_names, save_dir):
    """Plot fitness evolution for each model - Best only"""
    n_outputs = len(output_names)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for i, name in enumerate(output_names):
        model = models[name]
        ax = axes[i]

        best_fitness = model.run_details_['best_fitness']
        generations = range(len(best_fitness))

        ax.plot(generations, best_fitness, 'b-', linewidth=2.5, label='Best')

        ax.set_xlabel('Generation', fontsize=9, fontweight='bold')
        ax.set_ylabel('Fitness (MSE)', fontsize=9, fontweight='bold')
        ax.set_title(f'{name}\nFinal: {best_fitness[-1]:.6f} ({len(generations)} gen)',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.set_yscale('log')  # Log scale to see changes better

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evolution.png'), dpi=300, bbox_inches='tight')


def plot_comparison(y_test, y_pred, output_names, results, save_dir):
    """Plot predicted vs actual values for all outputs"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, name in enumerate(output_names):
        ax = axes[i]

        y_true = y_test[:, i]
        y_model = y_pred[:, i]

        # Scatter plot
        ax.scatter(y_true, y_model, alpha=0.4, s=8, edgecolors='none', c='blue')

        # Perfect prediction line
        min_val = min(y_true.min(), y_model.min())
        max_val = max(y_true.max(), y_model.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Prediction')

        # Labels and title
        ax.set_xlabel('Actual', fontsize=10, fontweight='bold')
        ax.set_ylabel('Predicted', fontsize=10, fontweight='bold')
        ax.set_title(f'{name}\nR² = {results[name]["r2"]:.4f}, RMSE = {results[name]["rmse"]:.4f}',
                     fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {os.path.join(save_dir, 'prediction_comparison.png')}")


def plot_residuals(y_test, y_pred, output_names, save_dir):
    """Plot residuals for all outputs"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, name in enumerate(output_names):
        ax = axes[i]

        y_true = y_test[:, i]
        y_model = y_pred[:, i]
        residuals = y_true - y_model

        # Residual scatter plot
        ax.scatter(y_model, residuals, alpha=0.4, s=8, edgecolors='none', c='blue')
        ax.axhline(y=0, color='r', linestyle='--', lw=2.5)

        # Labels and title
        ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
        ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=10, fontweight='bold')
        ax.set_title(f'{name} Residuals\nMean = {residuals.mean():.4e}, Std = {residuals.std():.4e}',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residuals.png'), dpi=300, bbox_inches='tight')
    print(f"Residual plot saved to {os.path.join(save_dir, 'residuals.png')}")


def print_equations(models, output_names, results):
    """Print discovered equations in readable format"""
    print("\n" + "=" * 80)
    print("DISCOVERED EQUATIONS")
    print("=" * 80 + "\n")

    feature_names = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']
    feature_desc = [
        'r_x', 'r_y', 'r_z',
        'sqrt(r_x² + r_y² + r_z²)',
        'r_x²', 'r_y²', 'r_z²',
        'r_x*r_y', 'r_x*r_z', 'r_y*r_z'
    ]

    for name in output_names:
        model = models[name]
        equation = str(model._program)

        # Replace feature names with descriptions
        for i, (feat_name, feat_desc) in enumerate(zip(feature_names, feature_desc)):
            equation = equation.replace(feat_name, feat_desc)

        print(f"{name}:")
        print(f"  {equation}")
        print(f"  R² = {results[name]['r2']:.4f}, RMSE = {results[name]['rmse']:.4f}")
        print(
            f"  Fitness: {model._program.fitness_:.6f}, Depth: {model._program.depth_}, Length: {model._program.length_}")
        print()

    print("=" * 80 + "\n")


def save_results(models, output_names, results, output_dir):
    """Save models and equations"""
    os.makedirs(output_dir, exist_ok=True)

    feature_names = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']
    feature_desc = [
        'r_x', 'r_y', 'r_z',
        'sqrt(r_x² + r_y² + r_z²)',
        'r_x²', 'r_y²', 'r_z²',
        'r_x*r_y', 'r_x*r_z', 'r_y*r_z'
    ]

    with open(os.path.join(output_dir, 'equations.txt'), 'w') as f:
        f.write("Discovered Equations\n")
        f.write("=" * 80 + "\n\n")

        for name in output_names:
            model = models[name]
            equation = str(model._program)

            # Replace feature names with descriptions
            for feat_name, feat_desc in zip(feature_names, feature_desc):
                equation = equation.replace(feat_name, feat_desc)

            f.write(f"{name}:\n")
            f.write(f"  {equation}\n")
            f.write(f"  Fitness: {model._program.fitness_:.6f}\n")
            f.write(f"  Depth: {model._program.depth_}\n")
            f.write(f"  Length: {model._program.length_}\n")
            f.write(f"  R²: {results[name]['r2']:.4f}\n")
            f.write(f"  RMSE: {results[name]['rmse']:.4f}\n")
            f.write("\n")

    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(output_dir, 'results.csv'))

    print(f"Results saved to {output_dir}")


def main():
    print("Loading data...")
    X, y, outputs = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Outputs: {outputs}")

    models = fit_models(X_train, y_train, outputs)
    results, y_pred = evaluate(models, X_test, y_test, outputs)

    print_equations(models, outputs, results)

    plot_evolution(models, outputs, OUT_DIR)
    plot_comparison(y_test, y_pred, outputs, results, OUT_DIR)
    plot_residuals(y_test, y_pred, outputs, OUT_DIR)
    save_results(models, outputs, results, OUT_DIR)

    plt.show()  # Show all plots at once

    return models, results


if __name__ == "__main__":
    models, results = main()