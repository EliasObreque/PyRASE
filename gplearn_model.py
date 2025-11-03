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

DATA_PATH = "./results/data/rect_prism_data_1000_sample_10000"
OUT_DIR = "../gplearn_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

COL_OUT_DRAG_F = ['Fx_drag', 'Fy_drag', 'Fz_drag']
COL_OUT_DRAG_T = ['Tx_drag', 'Ty_drag', 'Tz_drag']


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
    """Create physics-informed features"""
    r_x, r_y, r_z = X[:, 0], X[:, 1], X[:, 2]
    
    return np.column_stack([
        r_x, r_y, r_z,
        np.sqrt(r_x**2 + r_y**2 + r_z**2),
        r_x**2, r_y**2, r_z**2,
        r_x*r_y, r_x*r_z, r_y*r_z,
    ])


def create_regressor(config=None):
    """Create SymbolicRegressor with configuration"""
    default_config = {
        'population_size': 5000,
        'generations': 20,
        'tournament_size': 20,
        'stopping_criteria': 0.01,
        'const_range': (-1., 1.),
        'init_depth': (2, 6),
        'init_method': 'half and half',
        'function_set': ['add', 'sub', 'mul', 'div', 'sqrt', 'abs'],
        'metric': 'mse',
        'parsimony_coefficient': 0.001,
        'p_crossover': 0.7,
        'p_subtree_mutation': 0.1,
        'p_hoist_mutation': 0.05,
        'p_point_mutation': 0.1,
        'max_samples': 0.9,
        'verbose': 1,
        'n_jobs': 1,
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
    
    return results


def plot_evolution(models, output_names, save_dir):
    """Plot fitness evolution for each model"""
    n_outputs = len(output_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, name in enumerate(output_names):
        model = models[name]
        ax = axes[i]
        
        generations = range(len(model.run_details_['best_fitness']))
        ax.plot(generations, model.run_details_['best_fitness'], label='Best')
        ax.plot(generations, model.run_details_['average_fitness'], label='Average')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title(f'{name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_results(models, output_names, results, output_dir):
    """Save models and equations"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'equations.txt'), 'w') as f:
        f.write("Discovered Equations\n")
        f.write("=" * 80 + "\n\n")
        
        for name in output_names:
            model = models[name]
            f.write(f"{name}:\n")
            f.write(f"  Expression: {model._program}\n")
            f.write(f"  Fitness: {model._program.fitness_:.6f}\n")
            f.write(f"  Depth: {model._program.depth_}\n")
            f.write(f"  Length: {model._program.length_}\n")
            f.write(f"  R²: {results[name]['r2']:.4f}\n")
            f.write(f"  RMSE: {results[name]['rmse']:.4f}\n")
            f.write("\n")
    
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(output_dir, 'results.csv'))
    
    print(f"\nResults saved to {output_dir}")


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
    results = evaluate(models, X_test, y_test, outputs)
    
    plot_evolution(models, outputs, OUT_DIR)
    save_results(models, outputs, results, OUT_DIR)
    
    return models, results


if __name__ == "__main__":
    models, results = main()