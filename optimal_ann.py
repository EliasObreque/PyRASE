# -*- coding: utf-8 -*-
"""
Optimal ANN Hyperparameter Search with Individual Config Tracking
Created by: O Break
Version: 2.0 - Enhanced with individual folder tracking

Features:
- Individual folders for each configuration
- Comprehensive evaluation plots
- Training curves for each config
- ESP32 and pickle exports per config
- Final comparison plots
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

from core.ann_tools import (
    prepare_data_for_training,
    MLP,
    run_train_batch,
    get_predictions_unscaled
)

# ==========================
# CONFIGURATION
# ==========================

# Data paths
MAIN_FOLDER = "./results/data/"
DATA_PATH = MAIN_FOLDER + "rect_prism_data_1000_sample_10000"

# Extract base filename for output folder
BASE_NAME_FILE = Path(DATA_PATH).stem  # e.g., "rect_prism_data_1000_sample_10000"
OUT_DIR = f"./results/optimization/{BASE_NAME_FILE}/"
os.makedirs(OUT_DIR, exist_ok=True)

# Model type
MODEL_DRAG = 'drag_f'  # 'drag_f', 'drag_t', or 'drag'

# Threshold for filtering near-zero values
THRESHOLD_VALUE = None  # Set to 1e-4 if needed

# ==========================
# HYPERPARAMETER GRID
# ==========================

LAYERS_LIST = [3, 4, 5]  # Number of hidden layers
HIDDEN_LIST = [4, 6, 8]  # Neurons per hidden layer
ACT_LIST = ["relu", "tanh"]  # Activation functions
LR_LIST = [1e-2, 1e-3]  # Learning rates
BATCH_LIST = [64, 128]  # Batch sizes
SEEDS = [0, 1, 2]  # Random seeds for reproducibility

# Training parameters
EPOCHS = 100
PATIENCE = 10
WEIGHT_DECAY = 1e-4

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================
# LOAD DATA
# ==========================

print("Loading data...")
with open(DATA_PATH, "rb") as file_:
    data_mesh = pickle.load(file_)

sim_data = data_mesh.get("sim_data")


# ==========================
# HELPER FUNCTIONS
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
        f.write(f"// ANN Model for {MODEL_DRAG}\n")
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
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_and_save_predictions(model, test_loader, scaler, config_dir, model_type):
    """Evaluate model and save prediction plots"""

    # Get test data
    tX = test_loader.dataset.tensors[0]
    tY = test_loader.dataset.tensors[1]

    # Get predictions
    P_test, Y_test, X_test = get_predictions_unscaled(model, tX, tY, scaler)

    # Determine output names
    if 'f' in model_type:
        output_names = ['Fx', 'Fy', 'Fz']
    elif 't' in model_type:
        output_names = ['Tx', 'Ty', 'Tz']
    else:
        output_names = [f'Output_{i}' for i in range(P_test.shape[1])]

    # Create prediction plots
    n_outputs = P_test.shape[1]
    fig, axes = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 4))

    if n_outputs == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, output_names)):
        ax.scatter(Y_test[:, i], P_test[:, i], alpha=0.5, s=20)

        # Perfect prediction line
        min_val = min(Y_test[:, i].min(), P_test[:, i].min())
        max_val = max(Y_test[:, i].max(), P_test[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        ax.set_xlabel(f'True {name}', fontsize=11)
        ax.set_ylabel(f'Predicted {name}', fontsize=11)
        ax.set_title(f'{name} Predictions', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(config_dir, 'predictions.png')
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
        per_output_metrics.append({
            'output': output_names[i],
            'mse': float(mse_i),
            'mae': float(mae_i),
            'rmse': float(rmse_i)
        })

    return {
        'overall': {'mse': float(mse), 'mae': float(mae), 'rmse': float(rmse)},
        'per_output': per_output_metrics
    }


def train_single_config(config_dict, config_id, seed_idx, seed):
    """Train a single configuration with given seed"""

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Prepare data
    res_data = prepare_data_for_training(
        data_mesh,
        output_type=MODEL_DRAG,
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

    return model, val_loss, train_losses, val_losses, test_loader, scaler


# ==========================
# MAIN HYPERPARAMETER SEARCH
# ==========================

def run_comprehensive_search():
    """Run hyperparameter search with individual folder tracking"""

    print("\n" + "=" * 70)
    print("COMPREHENSIVE HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(f"Output directory: {OUT_DIR}")
    print(f"Model type: {MODEL_DRAG}")
    print("=" * 70)

    all_results = []
    config_id = 0

    # Calculate total configurations
    total_configs = len(LAYERS_LIST) * len(HIDDEN_LIST) * len(ACT_LIST) * len(LR_LIST) * len(BATCH_LIST)

    print(f"\nTotal unique configurations: {total_configs}")
    print(f"Seeds per configuration: {len(SEEDS)}")
    print(f"Total training runs: {total_configs * len(SEEDS)}\n")

    for act in ACT_LIST:
        for layers in LAYERS_LIST:
            for hidden in HIDDEN_LIST:
                for lr in LR_LIST:
                    for batch in BATCH_LIST:

                        config_id += 1

                        # Create configuration dictionary
                        config_dict = {
                            'activation': act,
                            'layers': layers,
                            'hidden': hidden,
                            'lr': lr,
                            'batch': batch
                        }

                        # Calculate approximate parameters
                        in_dim = 3
                        out_dim = 3
                        params = (in_dim * hidden + hidden) + \
                                 max(0, layers - 1) * (hidden * hidden + hidden) + \
                                 (hidden * out_dim + out_dim)

                        config_dict['parameters'] = params

                        print("\n" + "=" * 70)
                        print(f"CONFIG {config_id}/{total_configs}")
                        print("=" * 70)
                        print(f"Activation: {act}")
                        print(f"Layers: {layers}")
                        print(f"Hidden: {hidden}")
                        print(f"Learning rate: {lr}")
                        print(f"Batch size: {batch}")
                        print(f"Parameters: {params:,}")
                        print("=" * 70)

                        # Create config folder
                        config_dir = os.path.join(OUT_DIR, f"config_{config_id}")
                        os.makedirs(config_dir, exist_ok=True)

                        # Save config info
                        save_config_info(config_dir, config_dict, config_id)

                        # Train with multiple seeds
                        seed_losses = []
                        best_seed_idx = 0
                        best_seed_loss = float('inf')

                        for seed_idx, seed in enumerate(SEEDS):
                            print(f"\n--- Seed {seed_idx + 1}/{len(SEEDS)} (seed={seed}) ---")

                            model, val_loss, train_losses, val_losses, test_loader, scaler = \
                                train_single_config(config_dict, config_id, seed_idx, seed)

                            seed_losses.append(val_loss)
                            print(f"Validation loss: {val_loss:.6f}")

                            # Track best seed
                            if val_loss < best_seed_loss:
                                best_seed_loss = val_loss
                                best_seed_idx = seed_idx
                                best_model = model
                                best_scaler = scaler
                                best_test_loader = test_loader
                                best_train_losses = train_losses
                                best_val_losses = val_losses

                        # Save results for best seed
                        print(f"\nBest seed: {SEEDS[best_seed_idx]} (loss: {best_seed_loss:.6f})")

                        # Save training curves
                        curves_path = os.path.join(config_dir, 'training_curves.png')
                        save_training_curves(best_train_losses, best_val_losses, curves_path)

                        # Evaluate and save predictions
                        test_metrics = evaluate_and_save_predictions(
                            best_model, best_test_loader, best_scaler,
                            config_dir, MODEL_DRAG
                        )

                        # Export best model (ESP32 + pickle)
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

                            f.write("Test Set Performance (Best Seed):\n")
                            f.write("-" * 70 + "\n")
                            f.write(f"MSE:  {test_metrics['overall']['mse']:.6f}\n")
                            f.write(f"MAE:  {test_metrics['overall']['mae']:.6f}\n")
                            f.write(f"RMSE: {test_metrics['overall']['rmse']:.6f}\n\n")

                            f.write("Per-Output Metrics:\n")
                            f.write("-" * 70 + "\n")
                            for metric in test_metrics['per_output']:
                                f.write(f"{metric['output']}:\n")
                                f.write(f"  MSE:  {metric['mse']:.6f}\n")
                                f.write(f"  MAE:  {metric['mae']:.6f}\n")
                                f.write(f"  RMSE: {metric['rmse']:.6f}\n")

                        # Store summary
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
                            'best_seed': SEEDS[best_seed_idx]
                        }
                        all_results.append(result_summary)

                        print(f"\nConfig {config_id} completed!")
                        print(f"Results saved to: {config_dir}")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    return results_df


def create_comparison_plots(results_df, out_dir):
    """Create comprehensive comparison plots"""

    print("\n" + "=" * 70)
    print("CREATING COMPARISON PLOTS")
    print("=" * 70)

    # 1. Config ID vs Loss (for choosing best)
    plt.figure(figsize=(14, 6))
    x = results_df['config_id'].values
    y_mean = results_df['val_loss_mean'].values
    y_std = results_df['val_loss_std'].values

    plt.errorbar(x, y_mean, yerr=y_std, fmt='o', capsize=3, alpha=0.7,
                 markersize=5, linewidth=1)

    # Highlight best config
    best_idx = y_mean.argmin()
    best_config_id = x[best_idx]
    plt.scatter([best_config_id], [y_mean[best_idx]],
                color='red', s=200, marker='*', zorder=5,
                label=f'Best (Config {best_config_id})')

    plt.xlabel('Configuration ID', fontsize=12)
    plt.ylabel('Validation Loss (mean ± std)', fontsize=12)
    plt.title('All Configurations Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plot1_path = os.path.join(out_dir, 'all_configs_comparison.png')
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot1_path}")

    # 2. N_layers vs Loss
    plt.figure(figsize=(10, 6))

    for act in results_df['activation'].unique():
        subset = results_df[results_df['activation'] == act]

        # Group by layers and compute mean
        grouped = subset.groupby('layers')['val_loss_mean'].agg(['mean', 'std'])

        plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                     marker='o', label=act, capsize=5, markersize=8, linewidth=2)

    plt.xlabel('Number of Layers', fontsize=12)
    plt.ylabel('Validation Loss (mean ± std)', fontsize=12)
    plt.title('Effect of Network Depth', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, title='Activation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot2_path = os.path.join(out_dir, 'layers_vs_loss.png')
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot2_path}")

    # 3. N_neurons vs Loss
    plt.figure(figsize=(10, 6))

    for act in results_df['activation'].unique():
        subset = results_df[results_df['activation'] == act]

        # Group by hidden neurons and compute mean
        grouped = subset.groupby('hidden')['val_loss_mean'].agg(['mean', 'std'])

        plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                     marker='s', label=act, capsize=5, markersize=8, linewidth=2)

    plt.xlabel('Number of Neurons per Layer', fontsize=12)
    plt.ylabel('Validation Loss (mean ± std)', fontsize=12)
    plt.title('Effect of Network Width', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, title='Activation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot3_path = os.path.join(out_dir, 'neurons_vs_loss.png')
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot3_path}")

    # 4. Parameters vs Loss (Pareto plot)
    plt.figure(figsize=(10, 6))

    for act in results_df['activation'].unique():
        subset = results_df[results_df['activation'] == act]
        plt.scatter(subset['parameters'], subset['val_loss_mean'],
                    alpha=0.7, s=80, label=act)

    plt.xlabel('Number of Parameters', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Model Complexity vs Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, title='Activation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot4_path = os.path.join(out_dir, 'parameters_vs_loss.png')
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot4_path}")

    # 5. Learning rate effect
    plt.figure(figsize=(10, 6))

    for act in results_df['activation'].unique():
        subset = results_df[results_df['activation'] == act]
        grouped = subset.groupby('lr')['val_loss_mean'].agg(['mean', 'std'])

        plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                     marker='D', label=act, capsize=5, markersize=8, linewidth=2)

    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('Validation Loss (mean ± std)', fontsize=12)
    plt.title('Effect of Learning Rate', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.legend(fontsize=11, title='Activation')
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
    print("OPTIMAL ANN TRAINING WITH INDIVIDUAL CONFIG TRACKING")
    print("=" * 70)
    print(f"Data: {BASE_NAME_FILE}")
    print(f"Output: {OUT_DIR}")
    print(f"Model type: {MODEL_DRAG}")
    print("=" * 70)

    # Run hyperparameter search
    results_df = run_comprehensive_search()

    # Save results CSV
    csv_path = os.path.join(OUT_DIR, "all_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nAll results saved to: {csv_path}")

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
    print(f"\nAll results: {csv_path}")
    print(f"Best config folder: {OUT_DIR}/config_{int(best_config['config_id'])}/")
    print(f"Comparison plots: {OUT_DIR}/*.png")
    print("=" * 70)
    print("\n✓ All tasks completed successfully!")
    print("=" * 70)