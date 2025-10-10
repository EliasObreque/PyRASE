"""
Created by Elias Obreque
Date: 28/09/2025
email: els.obrq@gmail.com
"""
import os, json, math, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================
# Constants (edit here)
# ==========================
DATA_PATH = "../OneDrive_1_15-9-2025/ray_tracing_simulation_data.csv"  # CSV with v_x,v_y,v_z + 12 outputs
RHO       = 5e-13                # kg/m^3
V_MAG     = 7800.0               # m/s
P_SRP     = 4.57e-6              # N/m^2 (I/c at 1 AU)
PRUNE_AMT = 0.6                  # global unstructured sparsity (0..0.95)
EPOCHS    = 10                  # early-stop upper bound
PATIENCE  = 5
FINETUNE  = 20                   # after pruning
DEVICE    = "cpu"

print(torch.cuda.is_available())   # True if GPU is working
print(torch.version.cuda)          # Should print "12.1"
print(torch.cuda.get_device_name(0))

# Hyperparameter grids
LAYERS_LIST = [3, 4]
HIDDEN_LIST = [4, 8]#, 12, 16]#, 24, 32, 48, 64]
ACT_LIST    = ["relu", "tanh"]
LR_LIST     = [1e-3, 3e-3]
BATCH_LIST  = [64]
SEEDS       = [0, 1, 2]          # multiple restarts to reveal local minima

# Output artifacts
OUT_DIR = "../ann_search_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

COL_IN  = ["v_x","v_y","v_z"]
COL_OUT = ['f_x_d','f_y_d','f_z_d','t_x_d','t_y_d','t_z_d',
           'f_x_s','f_y_s','f_z_s','t_x_s','t_y_s','t_z_s']

# ==========================
# Data prep
# ==========================
def factor_targets(df, v_mag, rho, P):
    for c in ['f_x_d','f_y_d','f_z_d','t_x_d','t_y_d','t_z_d']:
        df[c] = df[c] * (2.0/(rho*(v_mag**2)))
    for c in ['f_x_s','f_y_s','f_z_s','t_x_s','t_y_s','t_z_s']:
        df[c] = df[c] * (1.0/P)
    return df

def zscale_outputs(df):
    means = df[COL_OUT].mean().values.astype(np.float32)
    stds  = df[COL_OUT].std().values.astype(np.float32)
    df[COL_OUT] = (df[COL_OUT] - means) / stds
    return df, means, stds

def make_loaders(df, batch, seed=42):
    train_df, tmp = train_test_split(df, test_size=0.30, random_state=seed)
    val_df, test_df = train_test_split(tmp, test_size=1/3, random_state=seed)
    def to_dl(split_df, shuffle=False):
        X = torch.tensor(split_df[COL_IN].values, dtype=torch.float32)
        Y = torch.tensor(split_df[COL_OUT].values, dtype=torch.float32)
        ds = TensorDataset(X,Y)
        return DataLoader(ds, batch_size=batch, shuffle=shuffle), X, Y
    train_loader, _, _ = to_dl(train_df, shuffle=True)
    val_loader,   vX, vY   = to_dl(val_df)
    test_loader,  tX, tY = to_dl(test_df)
    return train_loader, val_loader, test_loader, vX, vY, tX, tY

# ==========================
# Model / train / eval
# ==========================
class MLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=12, hidden=16, layers=2, activation="relu"):
        super().__init__()
        acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        Act = acts[activation]
        seq = []
        last = in_dim
        for _ in range(layers):
            seq += [nn.Linear(last, hidden), Act()]
            last = hidden
        seq += [nn.Linear(last, out_dim), nn.Tanh()]  # bounded head like your baseline
        self.net = nn.Sequential(*seq)
    def forward(self, x): return self.net(x)

def run_train(model, train_loader, val_loader, epochs, lr, weight_decay=0.0, patience=20, device="cpu"):
    model.to(device)
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = math.inf; best_state=None; no_improve=0
    for ep in tqdm(range(epochs)):
        model.train()
        for xb,yb in train_loader:
            xb=xb.to(device); yb=yb.to(device)
            pred = model(xb); loss = crit(pred,yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval()
        vtot=0.0; vn=0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb=xb.to(device); yb=yb.to(device)
                vtot += crit(model(xb), yb).item()*xb.size(0); vn += xb.size(0)
        val_loss = vtot/max(vn,1)
        if val_loss < best - 1e-9:
            best = val_loss; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model, best

def evaluate_unscale(model, X, Y, means, stds, device="cpu"):
    model.eval(); X=X.to(device); Y=Y.to(device)
    with torch.no_grad():
        P = model(X)
    means_t = torch.tensor(means, dtype=P.dtype, device=P.device).view(1,-1)
    stds_t  = torch.tensor(stds,  dtype=P.dtype, device=P.device).view(1,-1)
    P_real  = P*stds_t*10 + means_t
    Y_real  = Y*stds_t*10 + means_t
    mae = torch.mean(torch.abs(P_real - Y_real)).item()
    rmse = torch.sqrt(torch.mean((P_real - Y_real)**2)).item()
    return mae, rmse

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

# ==========================
# Search + plots
# ==========================
def main():
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

if __name__ == "__main__":
    main()