# -*- coding: utf-8 -*-
"""
Optimal ANN Hyperparameter Search with PARALLEL EXECUTION
Created by: O Break
Version: 3.0 - Minimal parallel wrapper (keeps original code intact)

Simply wraps the configuration loop in multiprocessing without changing any logic.
"""

import os
import json
import time
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count

from core.ann_tools import (
    prepare_data_for_training,
    MLP,
    run_train_batch,
    get_predictions_unscaled
)
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
# CONFIGURATION
# ==========================

# Data paths
MAIN_FOLDER = "./results/data/"
DATA_PATH = MAIN_FOLDER + "rect_prism_data_1000_sample_10000"

# Model type
PERTURBATION_STATE = 'srp_f'


# Extract base filename for output folder
BASE_NAME_FILE = Path(DATA_PATH).stem
OUT_DIR = f"./results/optimization/{BASE_NAME_FILE}/{PERTURBATION_STATE}"
os.makedirs(OUT_DIR, exist_ok=True)

import pickle
data = pickle.load(open(DATA_PATH, 'rb'))
print("Keys:", list(data.keys()))

# Threshold for filtering near-zero values
THRESHOLD_VALUE = None

# ==========================
# HYPERPARAMETER GRID
# ==========================

LAYERS_LIST = [3, 4, 5, 6, 7, 8]
HIDDEN_LIST = [3, 4, 5, 6, 7, 8]
ACT_LIST = ["relu", "tanh"]
LR_LIST = [1e-2, 1e-3]
BATCH_LIST = [64, 128]
SEEDS = [0, 1, 2]

# Training parameters
EPOCHS = 500
PATIENCE = 10
WEIGHT_DECAY = 1e-4

# Parallelization settings
N_WORKERS = max(1, int(cpu_count()-1))
print(f"Available CPUs: {cpu_count()}, Using: {N_WORKERS} workers")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================
# LOAD DATA (GLOBAL)
# ==========================

print("Loading data...")
with open(DATA_PATH, "rb") as file_:
    data_mesh = pickle.load(file_)


# ==========================
# HELPER FUNCTIONS (UNCHANGED FROM ORIGINAL)
# ==========================

def count_params(model):
    """Count total trainable parameters in model"""
    return sum(p.numel() for p in model.parameters())


def convert_to_serializable(obj):
    """Convert numpy arrays and types to JSON-serializable formats"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj


def save_config_info(config_dir, config_dict, config_id):
    """Save configuration information to text file"""
    info_path = os.path.join(config_dir, "config_info.txt")

    with open(info_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"CONFIGURATION {config_id}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Hyperparameters:\n")
        f.write("-" * 70 + "\n")
        for key, value in config_dict.items():
            f.write(f"{key:20s}: {value}\n")
        f.write("\n")

    return info_path


def export_model_for_esp32(model, scaler, filename, config):
    """Export model to ESP32 C array format"""

    in_dim = model.net[0].in_features
    out_dim = model.net[-1].out_features
    hidden = config['hidden']
    layers = config['layers']
    activation = config['activation']
    total_params = sum(p.numel() for p in model.parameters())

    with open(filename, 'w') as f:
        f.write(f"// ANN Model for {PERTURBATION_STATE}\n")
        f.write(f"// Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("//" + "=" * 60 + "\n\n")

        f.write("// MODEL ARCHITECTURE\n")
        f.write(f"// Input dimension: {in_dim}\n")
        f.write(f"// Output dimension: {out_dim}\n")
        f.write(f"// Hidden layers: {layers}\n")
        f.write(f"// Neurons per hidden layer: {hidden}\n")
        f.write(f"// Activation function: {activation}\n")
        f.write(f"// Total parameters: {total_params}\n\n")

        # Export weights and biases
        layer_idx = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                weights = param.data.cpu().numpy()
                rows, cols = weights.shape

                f.write(f"// Layer {layer_idx}: {name}\n")
                f.write(f"// Shape: ({rows}, {cols})\n")
                f.write(f"float W{layer_idx}[{rows}][{cols}] = {{\n")

                for i in range(rows):
                    f.write("  {")
                    f.write(", ".join([f"{weights[i, j]:.8f}" for j in range(cols)]))
                    f.write("}")
                    if i < rows - 1:
                        f.write(",")
                    f.write("\n")

                f.write("};\n\n")

            elif 'bias' in name:
                biases = param.data.cpu().numpy()
                size = len(biases)

                f.write(f"// Layer {layer_idx}: {name}\n")
                f.write(f"// Shape: ({size},)\n")
                f.write(f"float B{layer_idx}[{size}] = {{\n")
                f.write("  ")
                f.write(", ".join([f"{b:.8f}" for b in biases]))
                f.write("\n};\n\n")

                layer_idx += 1

        # Export scaler parameters
        f.write("\n// NORMALIZATION PARAMETERS\n")
        if hasattr(scaler, 'data_min_'):
            f.write("// Scaler type: MinMaxScaler\n\n")

            f.write(f"float input_min[{in_dim}] = {{")
            f.write(", ".join([f"{x:.8f}" for x in scaler.data_min_]))
            f.write("};\n\n")

            f.write(f"float input_max[{in_dim}] = {{")
            f.write(", ".join([f"{x:.8f}" for x in scaler.data_max_]))
            f.write("};\n\n")

            f.write(f"float scale[{in_dim}] = {{")
            f.write(", ".join([f"{x:.8f}" for x in scaler.scale_]))
            f.write("};\n\n")

            f.write(f"float min_[{in_dim}] = {{")
            f.write(", ".join([f"{x:.8f}" for x in scaler.min_]))
            f.write("};\n\n")

        f.write("// Usage: Implement forward pass with these weights\n")


def save_training_curves(train_losses, val_losses, save_path):
    """Save training curves plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2, alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', linewidth=2, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_and_save_predictions(model, tX, tY, scaler, config_dir, model_type, dataset_name='predictions'):
    """Evaluate model and save prediction plots with R² score"""

    model.eval()

    # Get predictions - tX and tY are already tensors
    # get_predictions_unscaled(model, X, Y, scaler, normalization='quantile', device=DEVICE)
    P_test, Y_test, X_test = get_predictions_unscaled(model, tX, tY, scaler, normalization='minmax', device=DEVICE)

    # Output names
    if 'f' in model_type:
        output_names = ['Fx', 'Fy', 'Fz']
    elif 't' in model_type:
        output_names = ['Tx', 'Ty', 'Tz']
    else:
        output_names = [f'Output_{i}' for i in range(P_test.shape[1])]

    # Compute R² score
    ss_res = np.sum((Y_test - P_test) ** 2)
    ss_tot = np.sum((Y_test - np.mean(Y_test, axis=0)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)

    # Create plots
    n_outputs = P_test.shape[1]
    fig, axes = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 4))

    if n_outputs == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, output_names)):
        ax.scatter(Y_test[:, i], P_test[:, i], alpha=0.5, s=20)

        min_val = min(Y_test[:, i].min(), P_test[:, i].min())
        max_val = max(Y_test[:, i].max(), P_test[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        # Compute per-output R²
        ss_res_i = np.sum((Y_test[:, i] - P_test[:, i]) ** 2)
        ss_tot_i = np.sum((Y_test[:, i] - np.mean(Y_test[:, i])) ** 2)
        r2_i = 1 - (ss_res_i / ss_tot_i)

        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name} (R²={r2_i:.4f})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Add main title for the entire figure
    dataset_title_map = {
        'train_predictions': 'Training Dataset',
        'val_predictions': 'Validation Dataset',
        'test_predictions': 'Test Dataset',
        'predictions': 'Predictions'
    }
    main_title = dataset_title_map.get(dataset_name, dataset_name.replace('_', ' ').title())
    fig.suptitle(main_title, fontweight='bold', y=1.02)
    
    plot_path = os.path.join(config_dir, f'{dataset_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Compute metrics
    mse = np.mean((P_test - Y_test) ** 2)
    mae = np.mean(np.abs(P_test - Y_test))
    rmse = np.sqrt(mse)

    # Per-output metrics
    per_output_metrics = []
    for i in range(n_outputs):
        mse_i = np.mean((P_test[:, i] - Y_test[:, i]) ** 2)
        mae_i = np.mean(np.abs(P_test[:, i] - Y_test[:, i]))
        rmse_i = np.sqrt(mse_i)
        
        # Per-output R²
        ss_res_i = np.sum((Y_test[:, i] - P_test[:, i]) ** 2)
        ss_tot_i = np.sum((Y_test[:, i] - np.mean(Y_test[:, i])) ** 2)
        r2_i = 1 - (ss_res_i / ss_tot_i)
        
        per_output_metrics.append({
            'output': output_names[i],
            'mse': float(mse_i),
            'mae': float(mae_i),
            'rmse': float(rmse_i),
            'r2': float(r2_i)
        })

    return {
        'overall': {'mse': float(mse), 'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2_score)},
        'per_output': per_output_metrics
    }


def train_single_config(config_dict, config_id, seed_idx, seed):
    """Train a single configuration with given seed (UNCHANGED FROM ORIGINAL)"""

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Prepare data
    res_data = prepare_data_for_training(
        data_mesh,
        output_type=PERTURBATION_STATE,
        batch_size=config_dict['batch'],
        seed=seed,
        normalization='minmax',
        threshold=THRESHOLD_VALUE
    )
    train_loader, val_loader, test_loader, vX, vY, tX, tY, scaler, col_out = res_data

    # Create model
    in_dim = len(train_loader.dataset.tensors[0][0])
    out_dim = len(train_loader.dataset.tensors[1][0])

    model = MLP(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden=config_dict['hidden'],
        layers=config_dict['layers'],
        activation=config_dict['activation']
    )

    # Train model
    model, val_loss, train_losses, val_losses = run_train_batch(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=config_dict['lr'],
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        device=DEVICE
    )

    # Extract training data tensors from train_loader
    train_X = train_loader.dataset.tensors[0]
    train_Y = train_loader.dataset.tensors[1]

    return model, val_loss, train_losses, val_losses, train_X, train_Y, vX, vY, tX, tY, scaler


# ==========================
# PARALLEL WRAPPER FUNCTION
# ==========================

def process_single_configuration(args):
    """
    Process one complete configuration (all seeds) - designed for parallel execution.
    This is the ONLY new function - it wraps the inner loop from the original code.
    """
    config_id, act, layers, hidden, lr, batch, params = args

    try:
        # Create configuration dictionary
        config_dict = {
            'activation': act,
            'layers': layers,
            'hidden': hidden,
            'lr': lr,
            'batch': batch,
            'parameters': params
        }

        # Create config folder
        config_dir = os.path.join(OUT_DIR, f"config_{config_id}")
        os.makedirs(config_dir, exist_ok=True)

        # Save config info
        save_config_info(config_dir, config_dict, config_id)

        # Train with multiple seeds (EXACT COPY FROM ORIGINAL)
        seed_losses = []
        best_seed_idx = 0
        best_seed_loss = float('inf')

        for seed_idx, seed in enumerate(SEEDS):
            model, val_loss, train_losses, val_losses, train_X, train_Y, vX, vY, tX, tY, scaler = \
                train_single_config(config_dict, config_id, seed_idx, seed)

            seed_losses.append(val_loss)

            # Track best seed
            if val_loss < best_seed_loss:
                best_seed_loss = val_loss
                best_seed_idx = seed_idx
                best_model = model
                best_scaler = scaler
                best_train_X = train_X
                best_train_Y = train_Y
                best_vX = vX
                best_vY = vY
                best_tX = tX
                best_tY = tY
                best_train_losses = train_losses
                best_val_losses = val_losses

        # Save results for best seed
        curves_path = os.path.join(config_dir, 'training_curves.png')
        save_training_curves(best_train_losses, best_val_losses, curves_path)

        # Evaluate on training, validation, and test sets
        train_metrics = evaluate_and_save_predictions(
            best_model, best_train_X, best_train_Y, best_scaler,
            config_dir, PERTURBATION_STATE, dataset_name='train_predictions'
        )

        val_metrics = evaluate_and_save_predictions(
            best_model, best_vX, best_vY, best_scaler,
            config_dir, PERTURBATION_STATE, dataset_name='val_predictions'
        )

        test_metrics = evaluate_and_save_predictions(
            best_model, best_tX, best_tY, best_scaler,
            config_dir, PERTURBATION_STATE, dataset_name='test_predictions'
        )

        # Export best model
        esp32_path = os.path.join(config_dir, 'model_esp32.txt')
        export_model_for_esp32(best_model, best_scaler, esp32_path, config_dict)

        pickle_path = os.path.join(config_dir, 'model.pkl')
        checkpoint = {
            'model_state_dict': best_model.state_dict(),
            'model_architecture': config_dict,
            'scaler': best_scaler,
            'test_metrics': test_metrics,
            'seed_losses': seed_losses,
            'best_seed': SEEDS[best_seed_idx],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(pickle_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        # Save metrics summary
        metrics_path = os.path.join(config_dir, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"CONFIGURATION {config_id} - RESULTS\n")
            f.write("=" * 70 + "\n\n")

            f.write("Validation Losses Across Seeds:\n")
            f.write("-" * 70 + "\n")
            for i, loss in enumerate(seed_losses):
                marker = " <-- BEST" if i == best_seed_idx else ""
                f.write(f"Seed {SEEDS[i]}: {loss:.6f}{marker}\n")
            f.write(f"\nMean: {np.mean(seed_losses):.6f}\n")
            f.write(f"Std:  {np.std(seed_losses):.6f}\n")
            f.write(f"Min:  {np.min(seed_losses):.6f}\n")
            f.write(f"Max:  {np.max(seed_losses):.6f}\n\n")

            f.write("Training Set Performance (Best Seed):\n")
            f.write("-" * 70 + "\n")
            f.write(f"MSE:  {train_metrics['overall']['mse']:.6f}\n")
            f.write(f"MAE:  {train_metrics['overall']['mae']:.6f}\n")
            f.write(f"RMSE: {train_metrics['overall']['rmse']:.6f}\n")
            f.write(f"R²:   {train_metrics['overall']['r2']:.6f}\n\n")

            f.write("Validation Set Performance (Best Seed):\n")
            f.write("-" * 70 + "\n")
            f.write(f"MSE:  {val_metrics['overall']['mse']:.6f}\n")
            f.write(f"MAE:  {val_metrics['overall']['mae']:.6f}\n")
            f.write(f"RMSE: {val_metrics['overall']['rmse']:.6f}\n")
            f.write(f"R²:   {val_metrics['overall']['r2']:.6f}\n\n")

            f.write("Test Set Performance (Best Seed):\n")
            f.write("-" * 70 + "\n")
            f.write(f"MSE:  {test_metrics['overall']['mse']:.6f}\n")
            f.write(f"MAE:  {test_metrics['overall']['mae']:.6f}\n")
            f.write(f"RMSE: {test_metrics['overall']['rmse']:.6f}\n")
            f.write(f"R²:   {test_metrics['overall']['r2']:.6f}\n\n")

            f.write("Per-Output Metrics (Test Set):\n")
            f.write("-" * 70 + "\n")
            for metric in test_metrics['per_output']:
                f.write(f"{metric['output']}:\n")
                f.write(f"  MSE:  {metric['mse']:.6f}\n")
                f.write(f"  MAE:  {metric['mae']:.6f}\n")
                f.write(f"  RMSE: {metric['rmse']:.6f}\n")
                f.write(f"  R²:   {metric['r2']:.6f}\n")

        # Return summary
        result_summary = {
            'config_id': config_id,
            'activation': act,
            'layers': layers,
            'hidden': hidden,
            'lr': lr,
            'batch': batch,
            'parameters': params,
            'val_loss_mean': float(np.mean(seed_losses)),
            'val_loss_std': float(np.std(seed_losses)),
            'val_loss_min': float(np.min(seed_losses)),
            'val_loss_max': float(np.max(seed_losses)),
            'test_mse': test_metrics['overall']['mse'],
            'test_mae': test_metrics['overall']['mae'],
            'test_rmse': test_metrics['overall']['rmse'],
            'test_r2': test_metrics['overall']['r2'],
            'best_seed': SEEDS[best_seed_idx]
        }

        return result_summary

    except Exception as e:
        print(f"ERROR in Config {config_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ==========================
# MAIN SEARCH (PARALLEL VERSION)
# ==========================

def run_comprehensive_search_parallel():
    """Run hyperparameter search with PARALLEL execution"""

    print("\n" + "=" * 70)
    print("PARALLEL HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(f"Output directory: {OUT_DIR}")
    print(f"Model type: {PERTURBATION_STATE}")
    print(f"Workers: {N_WORKERS}")
    print("=" * 70)

    # Generate all configurations
    configs_to_run = []
    config_id = 1  # Start from 1

    for act in ACT_LIST:
        for layers in LAYERS_LIST:
            for hidden in HIDDEN_LIST:
                for lr in LR_LIST:
                    for batch in BATCH_LIST:
                        # Calculate parameters
                        in_dim = 3
                        out_dim = 3
                        params = (in_dim * hidden + hidden) + \
                                 max(0, layers - 1) * (hidden * hidden + hidden) + \
                                 (hidden * out_dim + out_dim)

                        configs_to_run.append((config_id, act, layers, hidden, lr, batch, params))
                        config_id += 1

    total_configs = len(configs_to_run)
    print(f"\nTotal unique configurations: {total_configs}")
    print(f"Seeds per configuration: {len(SEEDS)}")
    print(f"Total training runs: {total_configs * len(SEEDS)}\n")

    # Run in parallel
    print("=" * 70)
    print("STARTING PARALLEL TRAINING")
    print("=" * 70)

    start_time = time.time()

    with Pool(processes=N_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_single_configuration, configs_to_run),
            total=total_configs,
            desc="Training configs",
            unit="config"
        ))

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"PARALLEL TRAINING COMPLETED in {elapsed_time / 60:.2f} minutes")
    print("=" * 70)

    # Filter out failed configs
    all_results = [r for r in results if r is not None]

    if len(all_results) < total_configs:
        print(f"\nWARNING: {total_configs - len(all_results)} configurations failed")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    return results_df


def create_comparison_plots(results_df, out_dir):
    """Create comprehensive comparison plots (UNCHANGED FROM ORIGINAL)"""

    print("\n" + "=" * 70)
    print("CREATING COMPARISON PLOTS")
    print("=" * 70)

    # 1. Config ID vs Loss
    plt.figure(figsize=(8, 6))
    x = results_df['config_id'].values
    y_mean = results_df['val_loss_mean'].values
    y_std = results_df['val_loss_std'].values

    plt.errorbar(x, y_mean, yerr=y_std, fmt='o', capsize=3, alpha=0.7,
                 markersize=5, linewidth=1)

    best_idx = y_mean.argmin()
    best_config_id = x[best_idx]
    plt.scatter([best_config_id], [y_mean[best_idx]],
                color='red', s=200, marker='*', zorder=5,
                label=f'Best (Config {best_config_id})')

    plt.xlabel('Configuration ID')
    plt.ylabel('Validation Loss (mean ± std)')
    plt.title('Configuration Performance Comparison', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot1_path = os.path.join(out_dir, 'all_configs_comparison.png')
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot1_path}")

    # 2. N_layers vs Loss
    plt.figure(figsize=(8, 6))

    for act in results_df['activation'].unique():
        subset = results_df[results_df['activation'] == act]
        grouped = subset.groupby('layers')['val_loss_mean'].agg(['mean', 'std'])
        plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                     marker='o', label=act, capsize=5, markersize=8, linewidth=2)

    plt.xlabel('Number of Layers')
    plt.ylabel('Validation Loss (mean ± std)')
    plt.title('Network Depth vs Performance', fontweight='bold')
    plt.legend(title='Activation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot2_path = os.path.join(out_dir, 'layers_vs_loss.png')
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot2_path}")

    # 3. N_neurons vs Loss
    plt.figure(figsize=(8, 6))

    for act in results_df['activation'].unique():
        subset = results_df[results_df['activation'] == act]
        grouped = subset.groupby('hidden')['val_loss_mean'].agg(['mean', 'std'])
        plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                     marker='s', label=act, capsize=5, markersize=8, linewidth=2)

    plt.xlabel('Number of Neurons per Layer')
    plt.ylabel('Validation Loss (mean ± std)')
    plt.title('Network Width vs Performance', fontweight='bold')
    plt.legend(title='Activation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot3_path = os.path.join(out_dir, 'neurons_vs_loss.png')
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot3_path}")

    # 4. Parameters vs Loss (Pareto plot)
    plt.figure(figsize=(8, 6))

    for act in results_df['activation'].unique():
        subset = results_df[results_df['activation'] == act]
        plt.scatter(subset['parameters'], subset['val_loss_mean'],
                    alpha=0.7, s=80, label=act)

    plt.xlabel('Number of Parameters')
    plt.ylabel('Validation Loss')
    plt.title('Model Complexity vs Performance', fontweight='bold')
    plt.legend(title='Activation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot4_path = os.path.join(out_dir, 'parameters_vs_loss.png')
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot4_path}")

    # 5. Learning rate effect
    plt.figure(figsize=(8, 6))

    for act in results_df['activation'].unique():
        subset = results_df[results_df['activation'] == act]
        grouped = subset.groupby('lr')['val_loss_mean'].agg(['mean', 'std'])
        plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                     marker='D', label=act, capsize=5, markersize=8, linewidth=2)

    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Loss (mean ± std)')
    plt.title('Learning Rate vs Performance', fontweight='bold')
    plt.xscale('log')
    plt.legend(title='Activation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot5_path = os.path.join(out_dir, 'lr_vs_loss.png')
    plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot5_path}")

    print("=" * 70)


# ==========================
# MAIN EXECUTION
# ==========================

if __name__ == "__main__":
    start_time = time.time()

    print("\n" + "=" * 70)
    print("OPTIMAL ANN TRAINING WITH PARALLEL EXECUTION")
    print("=" * 70)
    print(f"Data: {BASE_NAME_FILE}")
    print(f"Output: {OUT_DIR}")
    print(f"Model type: {PERTURBATION_STATE}")
    print("=" * 70)

    # Run parallel hyperparameter search
    # if file not exist
    csv_path = os.path.join(OUT_DIR, "all_results.csv")
    if not os.path.exists(csv_path) or True:
        results_df = run_comprehensive_search_parallel()
        results_df.to_csv(csv_path, index=False)
        print(f"\nAll results saved to: {csv_path}")
    else:
        results_df = pd.read_csv(csv_path)
        print("Keys:", list(results_df.keys()))
    # Save results CSV
  


    # Find and report best configuration
    best_idx = results_df['val_loss_mean'].idxmin()
    best_config = results_df.loc[best_idx]

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"Config ID: {best_config['config_id']}")
    print(f"Activation: {best_config['activation']}")
    print(f"Layers: {int(best_config['layers'])}")
    print(f"Hidden neurons: {int(best_config['hidden'])}")
    print(f"Learning rate: {best_config['lr']}")
    print(f"Batch size: {int(best_config['batch'])}")
    print(f"Parameters: {int(best_config['parameters']):,}")
    print(f"\nValidation loss: {best_config['val_loss_mean']:.6f} ± {best_config['val_loss_std']:.6f}")
    print(f"Test RMSE: {best_config['test_rmse']:.6f}")
    print(f"Test MAE: {best_config['test_mae']:.6f}")
    print(f"Test R²: {best_config['test_r2']:.6f}")
    print(f"\nBest model location: {OUT_DIR}/config_{int(best_config['config_id'])}/")
    print("=" * 70)

    # Save best config info
    best_info_path = os.path.join(OUT_DIR, "BEST_CONFIG_INFO.txt")
    with open(best_info_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BEST CONFIGURATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Config ID: {best_config['config_id']}\n")
        f.write(f"Folder: config_{int(best_config['config_id'])}/\n\n")
        f.write("Hyperparameters:\n")
        f.write(f"  Activation: {best_config['activation']}\n")
        f.write(f"  Layers: {int(best_config['layers'])}\n")
        f.write(f"  Hidden neurons: {int(best_config['hidden'])}\n")
        f.write(f"  Learning rate: {best_config['lr']}\n")
        f.write(f"  Batch size: {int(best_config['batch'])}\n")
        f.write(f"  Parameters: {int(best_config['parameters']):,}\n\n")
        f.write("Performance:\n")
        f.write(f"  Val loss: {best_config['val_loss_mean']:.6f} ± {best_config['val_loss_std']:.6f}\n")
        f.write(f"  Test MSE: {best_config['test_mse']:.6f}\n")
        f.write(f"  Test MAE: {best_config['test_mae']:.6f}\n")
        f.write(f"  Test RMSE: {best_config['test_rmse']:.6f}\n")
        f.write(f"  Test R²: {best_config['test_r2']:.6f}\n")

    # Create comparison plots
    create_comparison_plots(results_df, OUT_DIR)

    # Final summary
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total configurations tested: {len(results_df)}")
    print(f"Best config ID: {int(best_config['config_id'])}")
    print(f"Total elapsed time: {elapsed_time / 60:.2f} minutes")
    print(f"Parallel workers used: {N_WORKERS}")
    print(f"\nAll results: {csv_path}")
    print(f"Best config folder: {OUT_DIR}/config_{int(best_config['config_id'])}/")
    print(f"Comparison plots: {OUT_DIR}/*.png")
    print("=" * 70)