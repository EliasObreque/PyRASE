"""
Created by Elias Obreque
Date: 28/09/2025
email: els.obrq@gmail.com
"""
import os, json, math, time, random
import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer
from core.ann_tools import (prepare_data_for_training, MLP, run_train, run_train_batch,
                            evaluate_unscale, get_predictions_unscaled,
                            get_prediction_scaled)
from core.monitor import plot_predictions_by_axis



# Output artifacts
OUT_DIR = "../ann_search_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
    

# ==========================
# Constants 
# ==========================
DATA_PATH = "./results/data/rect_prism_data_1000_sample_10000"  # CSV with v_x,v_y,v_z + 12 outputs

with open(DATA_PATH, "rb") as file_:
    data_mesh = pickle.load(file_)

sim_data = data_mesh.get("sim_data")

# Usage
res_data = prepare_data_for_training(
    data_mesh, 
    output_type='drag_t', 
    batch_size=64, 
    seed=42,
    normalization='minmax'  # 'minmax' ([-1,1]), 'maxabs' (scale by max|x|), 'zscore' (standard), 'quantile' (uniform distribution), or 'none'
)
train_loader, val_loader, test_loader, vX, vY, tX, tY, scaler, col_out = res_data

# Get training data (already transformed by quantile_scale_outputs)
y_train_transformed = train_loader.dataset.tensors[1].cpu().numpy()

# Get original data for comparison
res_data_original = prepare_data_for_training(
    data_mesh, 
    output_type='drag_t', 
    batch_size=256, 
    seed=42,
    normalization='none'
)
y_train_original = res_data_original[0].dataset.tensors[1].cpu().numpy()

# Plot comparison
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

# Row 1: Original
axes[0, 0].hist(y_train_original[:, 0], bins=50, alpha=0.7, color='blue')
axes[0, 0].set_title("Original x-axis")
axes[0, 1].hist(y_train_original[:, 1], bins=50, alpha=0.7, color='green')
axes[0, 1].set_title("Original y-axis")
axes[0, 2].hist(y_train_original[:, 2], bins=50, alpha=0.7, color='red')
axes[0, 2].set_title("Original z-axis")

# Row 2: After Quantile Transform (should be uniform)
axes[1, 0].hist(y_train_transformed[:, 0], bins=50, alpha=0.7, color='blue')
axes[1, 0].set_title("Quantile x-axis")
axes[1, 1].hist(y_train_transformed[:, 1], bins=50, alpha=0.7, color='green')
axes[1, 1].set_title("Quantile y-axis")
axes[1, 2].hist(y_train_transformed[:, 2], bins=50, alpha=0.7, color='red')
axes[1, 2].set_title("Quantile z-axis")

plt.tight_layout()

# Recover data using the scaler from prepare_data_for_training
"""
fx_recovered = scaler.inverse_transform(y_train_transformed[:, 0:1])
fy_recovered = scaler.inverse_transform(y_train_transformed[:, 1:2])
fz_recovered = scaler.inverse_transform(y_train_transformed[:, 2:3])

# Plot recovered data
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].hist(fx_recovered, bins=50, alpha=0.7, color='blue')
axes[0].set_title("Recovered x-axis")

axes[1].hist(fy_recovered, bins=50, alpha=0.7, color='green')
axes[1].set_title("Recovered y-axis")

axes[2].hist(fz_recovered, bins=50, alpha=0.7, color='red')
axes[2].set_title("Recovered z-axis")

plt.tight_layout()
plt.show()

# Verify recovery accuracy
print(f"Fx recovery error: {np.abs(fx_recovered - y_train_original[:, 0:1]).max():.6f}")
print(f"Fy recovery error: {np.abs(fy_recovered - y_train_original[:, 1:2]).max():.6f}")
print(f"Fz recovery error: {np.abs(fz_recovered - y_train_original[:, 2:3]).max():.6f}")
"""
#%%

IN_DIM = 3      # Input features (R, P, Y)
OUT_DIM = 3    # Output features (12 joint angles)
HIDDEN = 6     # Hidden neurons layer size

LAYERS = 6      # Number of hidden layers
ACTIVATION = "gelu" # {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
EPOCHS = 10000
LR = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 1000


FINETUNE  = 20                   # after pruning
   
# Hyperparameter grids
LAYERS_LIST = [3, 4]
HIDDEN_LIST = [4, 8]#, 12, 16]#, 24, 32, 48, 64]
ACT_LIST    = ["relu", "tanh"]
LR_LIST     = [1e-3, 3e-3]
BATCH_LIST  = [64]
SEEDS       = [0, 1, 2]          # multiple restarts to reveal local minima

# Create and Train Model
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


#model, best_val_loss = run_train(
#    model=model,
#    train_loader=train_loader,
#    val_loader=val_loader,
#    epochs=EPOCHS,
#    lr=LR,
#    weight_decay=WEIGHT_DECAY,
#    patience=PATIENCE,
#)
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

# ==========================
# Evaluate on All Sets
# ==========================
print("\n" + "="*50)
print("Evaluation Results")
print("="*50)

#%%


# Get predictions for validation set
print("\nEvaluating validation set...")
t0 = time.time()
P_val, Y_val, X_val = get_predictions_unscaled(model, vX, vY, scaler)
print(time.time() - t0)
# Get predictions for test set
print("Evaluating test set...")
P_test, Y_test, X_test = get_predictions_unscaled(model, tX, tY, scaler)


plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.yscale("log")
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nFinal Results:")
print(f"Best Validation Loss: {best_val_loss:.6f}")

#%%

q_inf = sim_data["q_inf"]

P_np, Y_np, X_np = get_prediction_scaled(model, tX, tY, scaler)
    
print("\nGenerating validation set plots...")
fig_val_forces, fig_val_torques = plot_predictions_by_axis(
    P_np, Y_np, title_prefix="Validation Set")

#%%
P_train, Y_train, X_train = get_predictions_unscaled(model, train_loader.dataset.tensors[0], 
                                                       train_loader.dataset.tensors[1], scaler)

fig_val_forces, fig_val_torques = plot_predictions_by_axis(
    P_train, Y_train, title_prefix="Training Set")


"""


def count_params(model):
    return sum(p.numel() for p in model.parameters())

def global_prune(model, amount=0.6):
    from torch.nn.utils import prune
    params = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            params.append((m, "weight"))
    if params:
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
        for m in model.modules():
            if isinstance(m, nn.Linear) and hasattr(m, "weight_mask"):
                prune.remove(m, "weight")
    return model



if __name__ == "__main__":
    
    t0 = time.time()
    df = pd.read_csv(DATA_PATH)
    assert set(COL_IN+COL_OUT).issubset(df.columns), "CSV must contain: " + ",".join(COL_IN+COL_OUT)

    df = factor_targets(df.copy(), V_MAG, RHO, P_SRP)
    df, means, stds = zscale_outputs(df)

    results = []  # store all trials
    best_rec = None

    for act in ACT_LIST:
        for layers in LAYERS_LIST:
            for hidden in HIDDEN_LIST:
                for lr in LR_LIST:
                    for batch in BATCH_LIST:
                        seed_losses = []
                        for seed in SEEDS:
                            print(f"act:{act} - layers:{layers} - hidden:{hidden} - lr:{lr} - batch:{batch}")
                            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                            train_loader, val_loader, test_loader, vX, vY, tX, tY = make_loaders(df, batch, seed=seed)
                            model = MLP(hidden=hidden, layers=layers, activation=act)
                            model, _ = run_train(model, train_loader, val_loader, epochs=EPOCHS, lr=lr,
                                                 weight_decay=0.0, patience=PATIENCE, device=DEVICE)
                            mae, rmse = evaluate_unscale(model, vX, vY, means, stds, device=DEVICE)
                            seed_losses.append(rmse)
                        rec = dict(act=act, layers=layers, hidden=hidden, lr=lr, batch=batch,
                                   rmse_mean=float(np.mean(seed_losses)),
                                   rmse_std=float(np.std(seed_losses)),
                                   rmse_min=float(np.min(seed_losses)),
                                   rmse_max=float(np.max(seed_losses)))
                        results.append(rec)
                        if (best_rec is None) or (rec["rmse_mean"] < best_rec["rmse_mean"]):
                            best_rec = rec

    # Save grid results
    res_df = pd.DataFrame(results)
    res_csv = os.path.join(OUT_DIR, "grid_results.csv")
    res_df.to_csv(res_csv, index=False)

    # --- Plots ---
    # 1) Heatmaps (layers x hidden) per activation (min RMSE over lr,batch,seeds)
    for act in ACT_LIST:
        sub = res_df[res_df["act"]==act]
        # pivot by taking min over lr,batch and seeds (we used mean; take min of rmse_mean per cell)
        mat = np.full((len(LAYERS_LIST), len(HIDDEN_LIST)), np.nan)
        for i,L in enumerate(LAYERS_LIST):
            for j,H in enumerate(HIDDEN_LIST):
                cell = sub[(sub["layers"]==L) & (sub["hidden"]==H)]
                if len(cell)>0:
                    mat[i,j] = cell["rmse_mean"].min()
        plt.figure(figsize=(8,4.5))
        im = plt.imshow(mat, aspect="auto", origin="lower")
        plt.colorbar(im, label="Validation RMSE")
        plt.xticks(ticks=np.arange(len(HIDDEN_LIST)), labels=HIDDEN_LIST)
        plt.yticks(ticks=np.arange(len(LAYERS_LIST)), labels=LAYERS_LIST)
        plt.xlabel("Hidden units per layer")
        plt.ylabel("Number of layers")
        plt.title(f"RMSE heatmap (activation={act})")
        plt.tight_layout()
        outp = os.path.join(OUT_DIR, f"heatmap_{act}.png")
        plt.savefig(outp, dpi=200); plt.close()

    # 2) Pareto: params vs RMSE (min over lr/batch/seeds), color by act, marker by layers
    colors = {"relu":"C0","tanh":"C1","gelu":"C2"}
    markers = {1:"o",2:"s",3:"^",4:"D"}
    plt.figure(figsize=(7,4.5))
    for act in ACT_LIST:
        sub = res_df[res_df["act"]==act]
        grouped = sub.groupby(["layers","hidden"], as_index=False)["rmse_mean"].min()
        for _, row in grouped.iterrows():
            layers, hidden, rmse = int(row["layers"]), int(row["hidden"]), float(row["rmse_mean"])
            # approximate param count for MLP (bias included): (3*h + h) + (layers-1)*(h*h + h) + (h*12 + 12)
            h = hidden; L = layers
            params = (3*h + h) + max(0,L-1)*(h*h + h) + (h*12 + 12)
            plt.scatter(params, rmse, c=colors[act], marker=markers[L], label=f"{act}" if L==1 else None)
    plt.xlabel("Approx. parameters")
    plt.ylabel("Validation RMSE (min over lr,batch,seeds)")
    # Deduplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(), loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pareto_png = os.path.join(OUT_DIR, "pareto_params_rmse.png")
    plt.savefig(pareto_png, dpi=200); plt.close()

    # 3) Variability across seeds for the top-5 configs (local vs global indication)
    top5 = res_df.sort_values("rmse_mean").head(5).copy()
    plt.figure(figsize=(7,4.5))
    x = np.arange(len(top5))
    means = top5["rmse_mean"].values
    stds  = top5["rmse_std"].values
    plt.errorbar(x, means, yerr=stds, fmt="o", capsize=4)
    labels = [f'{r.act}, L{int(r.layers)}, H{int(r.hidden)}' for r in top5.itertuples()]
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Validation RMSE (mean ± std over seeds)")
    plt.title("Top-5 configs: variability across random seeds")
    plt.tight_layout()
    seeds_png = os.path.join(OUT_DIR, "top5_seed_variability.png")
    plt.savefig(seeds_png, dpi=200); plt.close()

    # --- Train best on train+val, prune, finetune, test ---
    best = best_rec
    # Re-split once with a fixed seed for reporting
    batch = BATCH_LIST[0]
    train_loader, val_loader, test_loader, vX, vY, tX, tY = make_loaders(df, batch, seed=123)
    # Merge train+val
    def merge_loader(tl, vl):
        Xs, Ys = [], []
        for xb,yb in tl: Xs.append(xb); Ys.append(yb)
        for xb,yb in vl: Xs.append(xb); Ys.append(yb)
        X = torch.cat(Xs, dim=0); Y = torch.cat(Ys, dim=0)
        ds = TensorDataset(X,Y)
        return DataLoader(ds, batch_size=batch, shuffle=True)
    trainval_loader = merge_loader(train_loader, val_loader)

    model = MLP(hidden=best["hidden"], layers=best["layers"], activation=best["act"])
    model, _ = run_train(model, trainval_loader, test_loader, epochs=EPOCHS, lr=LR_LIST[0],
                         weight_decay=0.0, patience=PATIENCE, device=DEVICE)
    mae_b, rmse_b = evaluate_unscale(model, tX, tY, means, stds, device=DEVICE)

    # Prune + finetune
    model = global_prune(model, amount=PRUNE_AMT)
    model, _ = run_train(model, trainval_loader, test_loader, epochs=FINETUNE, lr=LR_LIST[0],
                         weight_decay=0.0, patience=max(5, PATIENCE//2), device=DEVICE)
    mae_a, rmse_a = evaluate_unscale(model, tX, tY, means, stds, device=DEVICE)

    # Save checkpoint + report
    ckpt = {
        "state_dict": model.state_dict(),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "in_cols": COL_IN,
        "out_cols": COL_OUT,
        "best_hparams": best,
        "metrics": {"test_before_prune": {"mae": mae_b, "rmse": rmse_b},
                    "test_after_prune":  {"mae": mae_a, "rmse": rmse_a}},
        "search_summary_csv": res_csv
    }
    ckpt_path = os.path.join(OUT_DIR, "best_model_pruned.pt")
    torch.save(ckpt, ckpt_path)

    # Print a minimal text report
    report = {
        "best_hparams": best,
        "test_mae_before_prune": round(mae_b,6),
        "test_rmse_before_prune": round(rmse_b,6),
        "test_mae_after_prune": round(mae_a,6),
        "test_rmse_after_prune": round(rmse_a,6),
        "artifacts": {
            "grid_results_csv": res_csv,
            "heatmaps": [os.path.join(OUT_DIR, f"heatmap_{a}.png") for a in ACT_LIST],
            "pareto_png": pareto_png,
            "top5_seed_variability_png": seeds_png,
            "checkpoint": ckpt_path
        },
        "elapsed_s": round(time.time()-t0,2)
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
"""