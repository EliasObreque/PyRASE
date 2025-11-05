# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 14:43:42 2025

@author: mndc5
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

DATA_PATH = "./results/data/rect_prism_data_1000_sample_10000"
OUT_DIR = "../pysr_search_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

COL_OUT_DRAG_F = ['Fx_drag', 'Fy_drag', 'Fz_drag']
COL_OUT_DRAG_T = ['Tx_drag', 'Ty_drag', 'Tz_drag']

PYSR_CONFIG = {
    'binary_operators': ['+', '-', '*', '/', '^'],
    'unary_operators': ['square', 'sqrt', 'abs'],
    'niterations': 100,
    'populations': 30,
    'population_size': 50,
    'maxsize': 30,
    'maxdepth': 8,
    'constraints': {'^': (-1, 3)},
    'procs': 8,
    'verbosity': 1,
}


def load_data(data_path, outputs=None):
    """Load and prepare data"""
    with open(data_path, "rb") as f:
        data_mesh = pickle.load(f)
    
    sim_data = data_mesh["sim_data"]
    
    X = np.column_stack([
        [sim['v_x'] for sim in sim_data],
        [sim['v_y'] for sim in sim_data],
        [sim['v_z'] for sim in sim_data]
    ])
    
    if outputs is None:
        outputs = COL_OUT_DRAG_F + COL_OUT_DRAG_T
    
    y = np.column_stack([[sim[out] for sim in sim_data] for out in outputs])
    
    return X, y, outputs


def engineer_features(X):
    """Create physics-informed features"""
    v_x, v_y, v_z = X[:, 0], X[:, 1], X[:, 2]
    
    return np.column_stack([
        v_x, v_y, v_z,
        np.sqrt(v_x**2 + v_y**2 + v_z**2),
        v_x**2, v_y**2, v_z**2,
        v_x*v_y, v_x*v_z, v_y*v_z,
        np.abs(v_x), np.abs(v_y), np.abs(v_z)
    ]), ['v_x', 'v_y', 'v_z', 'v_mag', 'v_x_sq', 'v_y_sq', 'v_z_sq', 
         'v_xy', 'v_xz', 'v_yz', 'abs_vx', 'abs_vy', 'abs_vz']


def fit_pysr(X_train, y_train, output_names, config=None):
    """Fit PySR models for each output"""
    if config is None:
        config = PYSR_CONFIG
    
    X_eng, feature_names = engineer_features(X_train)
    models = {}
    
    for i, name in enumerate(output_names):
        print(f"\nFitting {name}...")
        model = PySRRegressor(**config, variable_names=feature_names)
        model.fit(X_eng, y_train[:, i])
        models[name] = model
        
        best = model.get_best()
        print(f"  Best: {best['equation']}")
        print(f"  Loss: {best['loss']:.6f}")
    
    return models


def predict_all(models, X, output_names):
    """Predict all outputs"""
    X_eng, _ = engineer_features(X)
    return np.column_stack([models[name].predict(X_eng) for name in output_names])


def evaluate(models, X_test, y_test, output_names):
    """Evaluate models"""
    y_pred = predict_all(models, X_test, output_names)
    
    print("\nTest Set Performance:")
    for i, name in enumerate(output_names):
        r2 = 1 - np.mean((y_test[:, i] - y_pred[:, i])**2) / np.var(y_test[:, i])
        print(f"{name}: R² = {r2:.4f}")


def save_results(models, output_names, output_dir):
    """Save all models and equations"""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model in models.items():
        model_dir = os.path.join(output_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        
        model.equations_.to_csv(os.path.join(model_dir, 'equations.csv'), index=False)
        model.save(os.path.join(model_dir, 'model.pkl'))
        
        with open(os.path.join(model_dir, 'best.txt'), 'w') as f:
            for _, row in model.equations_.nlargest(10, 'score').iterrows():
                f.write(f"{row['equation']}\n")
                f.write(f"  Complexity: {row['complexity']}, Loss: {row['loss']:.6f}\n\n")


def main():
    print("Loading data...")
    X, y, outputs = load_data(DATA_PATH)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)
    
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Outputs: {outputs}")
    
    models = fit_pysr(X_train, y_train, outputs)
    evaluate(models, X_test, y_test, outputs)
    save_results(models, outputs, OUT_DIR)
    
    print(f"\nResults saved to {OUT_DIR}")
    return models


if __name__ == "__main__":
    models = main()