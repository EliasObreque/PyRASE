# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 11:23:53 2025

@author: mndc5

Neural Network Training and ESP32 Export
"""

import os, json, math, time, random
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer
from core.ann_tools import (prepare_data_for_training, MLP, run_train, run_train_batch,
                            evaluate_unscale, get_predictions_unscaled,
                            get_prediction_scaled)
from core.monitor import plot_predictions_by_axis
import torch
import re

# Output directory
OUT_DIR = "../ann_search_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Constants
MAIN_FOLDER = "./results/data/"
DATA_PATH = MAIN_FOLDER + "rect_prism_data_1000_sample_10000"

# Load data
with open(DATA_PATH, "rb") as file_:
    data_mesh = pickle.load(file_)

sim_data = data_mesh.get("sim_data")
MODEL_DRAG = 'drag_t'

THRESHOLD_VALUE = None

# Prepare data
res_data = prepare_data_for_training(
    data_mesh,
    output_type=MODEL_DRAG,
    batch_size=64,
    seed=42,
    normalization='minmax',
    threshold=THRESHOLD_VALUE
)
train_loader, val_loader, test_loader, vX, vY, tX, tY, scaler, col_out = res_data

# Get training data (already transformed)
y_train_transformed = train_loader.dataset.tensors[1].cpu().numpy()

# Get original data for comparison
res_data_original = prepare_data_for_training(
    data_mesh,
    output_type=MODEL_DRAG,
    batch_size=int(256 / 2),
    seed=42,
    normalization='none',
    threshold=THRESHOLD_VALUE
)
y_train_original = res_data_original[0].dataset.tensors[1].cpu().numpy()

# Plot comparison
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
fig.suptitle(r"Drag Torque $\tilde{\boldsymbol{\tau}}_{d}$ distribution")
# Row 1: Original
axes[0, 0].hist(y_train_original[:, 0], bins=50, alpha=0.7, color='blue')
axes[0, 0].set_title("Original x-axis")
axes[0, 1].hist(y_train_original[:, 1], bins=50, alpha=0.7, color='green')
axes[0, 1].set_title("Original y-axis")
axes[0, 2].hist(y_train_original[:, 2], bins=50, alpha=0.7, color='red')
axes[0, 2].set_title("Original z-axis")

# Row 2: After transform
axes[1, 0].hist(y_train_transformed[:, 0], bins=50, alpha=0.7, color='blue')
axes[1, 0].set_title("Transformed x-axis")
axes[1, 1].hist(y_train_transformed[:, 1], bins=50, alpha=0.7, color='green')
axes[1, 1].set_title("Transformed y-axis")
axes[1, 2].hist(y_train_transformed[:, 2], bins=50, alpha=0.7, color='red')
axes[1, 2].set_title("Transformed z-axis")

plt.tight_layout()
plt.show()

# Recover data using scaler
recovered_data = scaler.inverse_transform(y_train_transformed)
fx_recovered = recovered_data[:, 0:1]
fy_recovered = recovered_data[:, 1:2]
fz_recovered = recovered_data[:, 2:3]

# Plot recovered data
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].hist(fx_recovered, bins=50, alpha=0.7, color='blue')
axes[0].set_title("Recovered x-axis")

axes[1].hist(fy_recovered, bins=50, alpha=0.7, color='green')
axes[1].set_title("Recovered y-axis")

axes[2].hist(fz_recovered, bins=50, alpha=0.7, color='red')
axes[2].set_title("Recovered z-axis")

plt.tight_layout()


# # Verify recovery accuracy
# print(f"Fx recovery error: {np.abs(fx_recovered - y_train_original[:, 0:1]).max():.6f}")
# print(f"Fy recovery error: {np.abs(fy_recovered - y_train_original[:, 1:2]).max():.6f}")
# print(f"Fz recovery error: {np.abs(fz_recovered - y_train_original[:, 2:3]).max():.6f}")
#
# print(f"Average values (x): {fx_recovered.mean()}")
# print(f"Average values (y): {fy_recovered.mean()}")
# print(f"Average values (z): {fz_recovered.mean()}")

# %%
# Model configuration
IN_DIM = 3
OUT_DIM = 3
HIDDEN = 6
LAYERS = 6
ACTIVATION = "tanh"
EPOCHS = 10000
LR = 0.01
WEIGHT_DECAY = 1e-4
PATIENCE = 1000
FINETUNE = 20

# Hyperparameter grids
LAYERS_LIST = [3, 4]
HIDDEN_LIST = [4, 8]
ACT_LIST = ["relu", "tanh"]
LR_LIST = [1e-3, 3e-3]
BATCH_LIST = [64]
SEEDS = [0, 1, 2]

# Create and train model
model = MLP(
    in_dim=IN_DIM,
    out_dim=OUT_DIM,
    hidden=HIDDEN,
    layers=LAYERS,
    activation=ACTIVATION
)

print(f"\nModel Architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

model, best_val_loss, train_losses, val_losses = run_train_batch(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=EPOCHS,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE
)

print(f"\nBest validation loss: {best_val_loss:.6f}")

# %%
# Evaluate on all sets
print("\n" + "=" * 50)
print("Evaluation Results")
print("=" * 50)

# Validation set
print("\nEvaluating validation set...")
t0 = time.time()
P_val, Y_val, X_val = get_predictions_unscaled(model, vX, vY, scaler)
print(f"Validation evaluation time: {time.time() - t0:.2f}s")

# Plot training curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.yscale("log")
plt.savefig(MAIN_FOLDER + 'training_curves.png', dpi=300, bbox_inches='tight')

print(f"\nFinal Results:")
print(f"Best Validation Loss: {best_val_loss:.6f}")

# %%
q_inf = sim_data["q_inf"]

# Test set evaluation - CORRECTED
print("\nEvaluating test set...")
P_test, Y_test, X_test = get_predictions_unscaled(model, tX, tY, scaler)

if "t" in MODEL_DRAG:
    is_force = False
    is_torque = True
elif "f" in MODEL_DRAG:
    is_force = True
    is_torque = False
else:
    is_force = True
    is_torque = True

print("\nGenerating test set plots...")
fig_test_forces, fig_test_torques = plot_predictions_by_axis(
    P_test, Y_test, is_force, is_torque, title_prefix="Test Set")

if fig_test_torques:
    fig_test_torques.savefig(MAIN_FOLDER + "test_torques.png", dpi=300, bbox_inches='tight')
if fig_test_forces:
    fig_test_forces.savefig(MAIN_FOLDER + "test_forces.png", dpi=300, bbox_inches='tight')

# Training set evaluation
print("\nEvaluating training set...")
P_train, Y_train, X_train = get_predictions_unscaled(model, train_loader.dataset.tensors[0],
                                                     train_loader.dataset.tensors[1], scaler)

print("\nGenerating training set plots...")
fig_train_forces, fig_train_torques = plot_predictions_by_axis(
    P_train, Y_train, is_force, is_torque, title_prefix="Training Set")

if fig_train_torques:
    fig_train_torques.savefig(MAIN_FOLDER + "train_torques.png", dpi=300, bbox_inches='tight')
if fig_train_forces:
    fig_train_forces.savefig(MAIN_FOLDER + "train_forces.png", dpi=300, bbox_inches='tight')

plt.show()


# %%
# Export model for ESP32
def export_model_for_esp32(model, scaler, filename, description="Neural Network Model"):
    """
    Export neural network weights and biases to text file for ESP32 deployment.
    """

    with open(filename, 'w') as f:
        f.write(f"// {description}\n")
        f.write(f"// Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("//" + "=" * 60 + "\n\n")

        # Model architecture
        f.write("// MODEL ARCHITECTURE\n")
        f.write(f"// Input dimension: {IN_DIM}\n")
        f.write(f"// Output dimension: {OUT_DIM}\n")
        f.write(f"// Hidden layers: {LAYERS}\n")
        f.write(f"// Neurons per hidden layer: {HIDDEN}\n")
        f.write(f"// Activation function: {ACTIVATION}\n")
        f.write(f"// Total parameters: {sum(p.numel() for p in model.parameters()):,}\n\n")

        # Extract weights and biases
        layer_idx = 0
        for name, param in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                weight_np = param.detach().cpu().numpy()

                f.write(f"// Layer {layer_idx}: {name}\n")
                f.write(f"// Shape: {weight_np.shape}\n")

                if 'weight' in name:
                    f.write(f"float W{layer_idx}[{weight_np.shape[0]}][{weight_np.shape[1]}] = {{\n")
                    for i in range(weight_np.shape[0]):
                        f.write("  {")
                        for j in range(weight_np.shape[1]):
                            f.write(f"{weight_np[i, j]:.8f}")
                            if j < weight_np.shape[1] - 1:
                                f.write(", ")
                        f.write("}")
                        if i < weight_np.shape[0] - 1:
                            f.write(",")
                        f.write("\n")
                    f.write("};\n\n")

                elif 'bias' in name:
                    f.write(f"float B{layer_idx}[{weight_np.shape[0]}] = {{\n  ")
                    for i in range(weight_np.shape[0]):
                        f.write(f"{weight_np[i]:.8f}")
                        if i < weight_np.shape[0] - 1:
                            f.write(", ")
                    f.write("\n};\n\n")
                    layer_idx += 1

        # Normalization parameters
        if scaler is not None:
            f.write("\n// NORMALIZATION PARAMETERS\n")
            f.write(f"// Scaler type: {type(scaler).__name__}\n")

            if hasattr(scaler, 'data_min_'):
                f.write(f"\n// Input min values\n")
                f.write(f"float input_min[{len(scaler.data_min_)}] = {{")
                f.write(", ".join([f"{x:.8f}" for x in scaler.data_min_]))
                f.write("};\n")

            if hasattr(scaler, 'data_max_'):
                f.write(f"\n// Input max values\n")
                f.write(f"float input_max[{len(scaler.data_max_)}] = {{")
                f.write(", ".join([f"{x:.8f}" for x in scaler.data_max_]))
                f.write("};\n")

            if hasattr(scaler, 'scale_'):
                f.write(f"\n// Scale factors\n")
                f.write(f"float scale[{len(scaler.scale_)}] = {{")
                f.write(", ".join([f"{x:.8f}" for x in scaler.scale_]))
                f.write("};\n")

            if hasattr(scaler, 'min_'):
                f.write(f"\n// Min values\n")
                f.write(f"float min_[{len(scaler.min_)}] = {{")
                f.write(", ".join([f"{x:.8f}" for x in scaler.min_]))
                f.write("};\n")

        # Usage instructions
        f.write("\n\n// USAGE INSTRUCTIONS FOR ESP32\n")
        f.write("// 1. Copy weight matrices (W0, W1, ...) and bias vectors (B0, B1, ...)\n")
        f.write("// 2. Implement forward pass:\n")
        f.write("//    - Apply normalization to inputs\n")
        f.write("//    - For each layer: output = activation(W * input + B)\n")
        f.write(f"//    - Use '{ACTIVATION}' activation function\n")
        f.write("//    - Apply inverse normalization to outputs if needed\n")
        f.write("// 3. Memory optimization: Use PROGMEM for weights\n")

    print(f"\nModel exported to: {filename}")
    print(f"File size: {os.path.getsize(filename) / 1024:.2f} KB")


def parse_exported_weights(filename):
    """Parse exported weights file and reconstruct model"""
    weights = []
    biases = []

    with open(filename, 'r') as f:
        content = f.read()

    # Extract weight matrices
    weight_pattern = r'float W(\d+)\[(\d+)\]\[(\d+)\] = \{([^}]+)\};'
    weight_matches = re.findall(weight_pattern, content, re.DOTALL)

    for idx, rows, cols, data in weight_matches:
        rows_data = []
        for row in data.split('},'):
            row = row.strip().replace('{', '').replace('}', '')
            if row:
                values = [float(x.strip()) for x in row.split(',') if x.strip()]
                if values:
                    rows_data.append(values)

        weight_matrix = np.array(rows_data)
        weights.append(weight_matrix)

    # Extract bias vectors
    bias_pattern = r'float B(\d+)\[(\d+)\] = \{([^}]+)\};'
    bias_matches = re.findall(bias_pattern, content, re.DOTALL)

    for idx, size, data in bias_matches:
        values = [float(x.strip()) for x in data.split(',') if x.strip()]
        bias_vector = np.array(values)
        biases.append(bias_vector)

    return weights, biases


def numpy_forward_pass(input_data, weights, biases, activation='tanh'):
    """Perform forward pass using numpy (simulating ESP32 behavior)"""
    x = input_data.copy()

    # Activation function
    if activation == 'tanh':
        act_func = np.tanh
    elif activation == 'relu':
        act_func = lambda x: np.maximum(0, x)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # Forward through all layers
    for i in range(len(weights)):
        x = np.dot(weights[i], x) + biases[i]

        # Apply activation (except last layer)
        if i < len(weights) - 1:
            x = act_func(x)

    return x


def verify_export(pytorch_model, exported_file, test_input, scaler=None, activation='tanh'):
    """Verify that exported weights produce same results as PyTorch model"""

    print("\n" + "=" * 60)
    print("VERIFICATION: Exported Weights vs PyTorch Model")
    print("=" * 60)

    # Parse exported weights
    print("\nParsing exported weights...")
    weights, biases = parse_exported_weights(exported_file)

    # Get PyTorch prediction (move model to CPU for verification)
    print("Running PyTorch model...")
    pytorch_model.eval()
    pytorch_model.cpu()  # Move model to CPU
    with torch.no_grad():
        input_tensor = torch.FloatTensor(test_input)
        pytorch_output = pytorch_model(input_tensor).cpu().numpy()

    print(f"PyTorch output: {pytorch_output}")

    # Get numpy prediction (simulating ESP32)
    print("Running numpy simulation (ESP32 equivalent)...")
    numpy_output = numpy_forward_pass(test_input[0], weights, biases, activation)

    print(f"Numpy output: {numpy_output}")

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    difference = np.abs(pytorch_output[0] - numpy_output)
    max_diff = np.max(difference)
    mean_diff = np.mean(difference)

    print(f"Maximum difference: {max_diff:.10f}")
    print(f"Mean difference: {mean_diff:.10f}")
    print(f"Relative error: {max_diff / (np.abs(pytorch_output[0]).max() + 1e-8) * 100:.6f}%")

    # Tolerance check
    tolerance = 1e-6
    if max_diff < tolerance:
        print(f"\nPASS: Differences below tolerance ({tolerance})")
        print("Export is correct. ESP32 implementation will work.")
    else:
        print(f"\nFAIL: Differences above tolerance ({tolerance})")
        print("Check export function or numpy implementation.")

    print("=" * 60)

    return max_diff < tolerance


def analyze_model_memory(weights, biases):
    """Analyze memory requirements for ESP32"""
    print("\n" + "=" * 60)
    print("MEMORY ANALYSIS FOR ESP32")
    print("=" * 60)

    # Count parameters
    total_params = 0
    for w, b in zip(weights, biases):
        total_params += w.size + b.size

    # Memory usage
    float32_size = total_params * 4
    float16_size = total_params * 2
    int8_size = total_params * 1

    print(f"\nTotal parameters: {total_params:,}")
    print(f"\nMemory requirements:")
    print(f"  float32: {float32_size:,} bytes ({float32_size / 1024:.2f} KB)")
    print(f"  float16: {float16_size:,} bytes ({float16_size / 1024:.2f} KB)")
    print(f"  int8:    {int8_size:,} bytes ({int8_size / 1024:.2f} KB)")

    # Activation memory (worst case: all neurons active)
    max_neurons = np.max([w.shape[0] for w in weights])
    activation_mem = max_neurons * 4 * 2

    print(f"\nActivation memory (runtime):")
    print(f"  {activation_mem:,} bytes ({activation_mem / 1024:.2f} KB)")

    print(f"\nTotal runtime memory (float32):")
    total_mem = float32_size + activation_mem
    print(f"  {total_mem:,} bytes ({total_mem / 1024:.2f} KB)")

    # ESP32 available memory
    esp32_sram = 520 * 1024
    usage_percent = (total_mem / esp32_sram) * 100

    print(f"\nESP32 SRAM usage: {usage_percent:.2f}%")

    if usage_percent < 10:
        print("Very comfortable fit - plenty of room for other code")
    elif usage_percent < 30:
        print("Good fit - sufficient room for application logic")
    elif usage_percent < 60:
        print("Moderate fit - be careful with other memory allocations")
    else:
        print("Tight fit - consider model compression or using PSRAM")

    print("=" * 60)


def compute_error_metrics(predictions, targets, labels=None):
    """Compute MSE, MAE, RMSE for predictions"""
    if labels is None:
        labels = [f"Output_{i}" for i in range(predictions.shape[1])]

    print("\n" + "=" * 60)
    print("Error Metrics")
    print("=" * 60)
    print(f"{'Metric':<15} {'MSE':<15} {'MAE':<15} {'RMSE':<15}")
    print("-" * 60)

    for i, label in enumerate(labels):
        mse = np.mean((predictions[:, i] - targets[:, i]) ** 2)
        mae = np.mean(np.abs(predictions[:, i] - targets[:, i]))
        rmse = np.sqrt(mse)
        print(f"{label:<15} {mse:<15.6f} {mae:<15.6f} {rmse:<15.6f}")

    # Overall metrics
    mse_overall = np.mean((predictions - targets) ** 2)
    mae_overall = np.mean(np.abs(predictions - targets))
    rmse_overall = np.sqrt(mse_overall)

    print("-" * 60)
    print(f"{'Overall':<15} {mse_overall:<15.6f} {mae_overall:<15.6f} {rmse_overall:<15.6f}")
    print("=" * 60)


# Export the trained model
export_filename = MAIN_FOLDER + "model_weights_esp32.txt"
export_model_for_esp32(model, scaler, export_filename,
                       description=f"ANN Model for Drag Force Prediction ({MODEL_DRAG})")

# Verify export
print("\n" + "="*60)
print("VERIFYING EXPORTED WEIGHTS")
print("="*60)

# Test with sample input
test_input = np.array([[1.0, 0.5, -0.2]])

# Verify that exported weights match PyTorch model
is_correct = verify_export(
    pytorch_model=model,
    exported_file=export_filename,
    test_input=test_input,
    scaler=scaler,
    activation=ACTIVATION
)

if is_correct:
    print("\nExport verification PASSED")
else:
    print("\nExport verification FAILED - check export function")

# Analyze memory requirements for ESP32
weights_exported, biases_exported = parse_exported_weights(export_filename)
analyze_model_memory(weights_exported, biases_exported)

# %%
# Summary statistics
print("\n### TRAINING SET ###")
compute_error_metrics(P_train, Y_train, labels=['Fx', 'Fy', 'Fz'])

print("\n### VALIDATION SET ###")
compute_error_metrics(P_val, Y_val, labels=['Fx', 'Fy', 'Fz'])

print("\n### TEST SET ###")
compute_error_metrics(P_test, Y_test, labels=['Fx', 'Fy', 'Fz'])

print("\n" + "="*60)
print("Training Complete")
print("="*60)