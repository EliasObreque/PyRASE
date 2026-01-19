# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 12:11:34 2025

@author: mndc5
"""
import pickle

import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler



print(torch.cuda.is_available())   # True if GPU is working
cuda_info = torch.version.cuda
print(cuda_info)          # Should print "12.1"
if cuda_info:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Define columns
COL_IN = ['r_x', 'r_y', 'r_z']
# COL_IN += ['r_xy', 'r_xz', 'r_yz']
COL_OUT_DRAG_F = ['Fx_drag', 'Fy_drag', 'Fz_drag']
COL_OUT_DRAG_T = ['Tx_drag', 'Ty_drag', 'Tz_drag']
COL_OUT_DRAG = COL_OUT_DRAG_F + COL_OUT_DRAG_T

COL_OUT_SRP_F = ['Fx_srp', 'Fy_srp', 'Fz_srp']
COL_OUT_SRP_T = ['Tx_srp', 'Ty_srp', 'Tz_srp']
COL_OUT_SRP = COL_OUT_SRP_F + COL_OUT_SRP_T
COL_OUT_ALL = COL_OUT_DRAG_F + COL_OUT_DRAG_T + COL_OUT_SRP



def prepare_dataframe(data_mesh):
    """Convert data_mesh dict to DataFrame using numpy arrays"""
    
    # Convert to numpy arrays
    r_array = np.array(data_mesh['r_'])
    F_drag_array = np.array(data_mesh['F_drag'])
    T_drag_array = np.array(data_mesh['T_drag'])
    F_srp_array = np.array(data_mesh['F_srp'])
    T_srp_array = np.array(data_mesh['T_srp'])
    
    df_dict = {
        # Input: direction vector (3 components)
        'r_x': r_array[:, 0],
        'r_y': r_array[:, 1],
        'r_z': r_array[:, 2],
        'r_xy': r_array[:, 0] * r_array[:, 1],
        'r_xz': r_array[:, 0] * r_array[:, 2],
        'r_yz': r_array[:, 1] * r_array[:, 2],

        # Outputs: Drag forces and torques
        'Fx_drag': F_drag_array[:, 0],
        'Fy_drag': F_drag_array[:, 1],
        'Fz_drag': F_drag_array[:, 2],
        'Tx_drag': T_drag_array[:, 0],
        'Ty_drag': T_drag_array[:, 1],
        'Tz_drag': T_drag_array[:, 2],
        
        # Outputs: SRP forces and torques
        'Fx_srp': F_srp_array[:, 0],
        'Fy_srp': F_srp_array[:, 1],
        'Fz_srp': F_srp_array[:, 2],
        'Tx_srp': T_srp_array[:, 0],
        'Ty_srp': T_srp_array[:, 1],
        'Tz_srp': T_srp_array[:, 2],
    }
    
    return pd.DataFrame(df_dict)

def factor_targets(df, sim_data):
    """
    Non-dimensionalize forces and torques
    
    Drag: F̂_d = F_d / q_inf,  τ̂_d = τ_d / q_inf
    SRP:  F̂_s = F_s / P_SRP,  τ̂_s = τ_s / P_SRP
    """
    q_inf = sim_data['q_inf']
    P_srp = sim_data['P_srp']
    
    # Non-dimensionalize DRAG
    for c in ['Fx_drag', 'Fy_drag', 'Fz_drag', 'Tx_drag', 'Ty_drag', 'Tz_drag']:
        df[c] = df[c] / q_inf
        
    for c in ['Tx_drag', 'Ty_drag', 'Tz_drag']:
        df[c] = df[c]
    
    # Non-dimensionalize SRP
    for c in ['Fx_srp', 'Fy_srp', 'Fz_srp', 'Tx_srp', 'Ty_srp', 'Tz_srp']:
        df[c] = df[c] / P_srp
    
    return df


def zscale_outputs(train_df, val_df, test_df, col_out):
    scaler = StandardScaler()
    scaler.fit(train_df[col_out])
    
    train_df[col_out] = scaler.transform(train_df[col_out])
    val_df[col_out] = scaler.transform(val_df[col_out])
    test_df[col_out] = scaler.transform(test_df[col_out])
    
    return train_df, val_df, test_df, scaler


def minmax_scale_outputs(train_df, val_df, test_df, col_out, feature_range=(-1, 1)):
    """
    Scale outputs using sklearn MinMaxScaler
    
    Args:
        train_df, val_df, test_df: DataFrames to scale
        col_out: List of output column names
        feature_range: Target range (default: (-1, 1))
    
    Returns:
        train_df, val_df, test_df (normalized), scaler object
    """
    # Initialize scaler
    feature_range = (-1, 1)
    scaler = MinMaxScaler(feature_range=feature_range)
    
    # Fit on training data only
    scaler.fit(train_df[col_out])
    
    # Transform all datasets
    train_df[col_out] = scaler.transform(train_df[col_out])
    val_df[col_out] = scaler.transform(val_df[col_out])
    test_df[col_out] = scaler.transform(test_df[col_out])
    
    return train_df, val_df, test_df, scaler


def make_loaders(train_df, val_df, test_df, col_in, col_out, batch):
    """Create DataLoaders"""
    def to_dl(split_df, shuffle=False):
        X = torch.tensor(split_df[col_in].values, dtype=torch.float32)
        Y = torch.tensor(split_df[col_out].values, dtype=torch.float32)
        ds = TensorDataset(X, Y)
        return DataLoader(ds, batch_size=batch, shuffle=shuffle), X, Y
    
    train_loader, _, _ = to_dl(train_df, shuffle=True)
    val_loader, vX, vY = to_dl(val_df)
    test_loader, tX, tY = to_dl(test_df)
    
    return train_loader, val_loader, test_loader, vX, vY, tX, tY




def maxabs_scale_outputs(train_df, val_df, test_df, col_out):
    """
    Scale outputs by dividing by max(abs(value)) from training set
    This scales to approximately [-1, 1] while preserving zero
    
    Returns:
        train_df, val_df, test_df (normalized), max_abs_vals
    """
    # Get max absolute value from training data only
    max_abs_vals = train_df[col_out].abs().max().values
    
    # Avoid division by zero
    max_abs_vals = np.where(max_abs_vals < 1e-10, 1.0, max_abs_vals)
    
    # Scale by dividing by max absolute value
    for df in [train_df, val_df, test_df]:
        df[col_out] = df[col_out] / max_abs_vals
    
    return train_df, val_df, test_df, max_abs_vals


def quantile_scale_outputs(train_df, val_df, test_df, col_out):
    """
    Scale outputs using QuantileTransformer to uniform distribution
    Each output gets equal representation regardless of frequency
    
    Args:
        train_df, val_df, test_df: DataFrames to scale
        col_out: List of output column names
    
    Returns:
        train_df, val_df, test_df (normalized), dict of scalers
    """
    from sklearn.preprocessing import QuantileTransformer
    
    # Store a transformer for each output column
    scalers = {}
    
    for col in col_out:
        # Fit transformer on training data only
        qt = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
        qt.fit(train_df[[col]])
        scalers[col] = qt
        
        # Transform all datasets
        train_df[col] = qt.transform(train_df[[col]]).flatten()
        val_df[col] = qt.transform(val_df[[col]]).flatten()
        test_df[col] = qt.transform(test_df[[col]]).flatten()
    
    return train_df, val_df, test_df, scalers


def prepare_data_for_training(data_mesh, output_type='drag', batch_size=32, seed=42, 
                               normalization='minmax', threshold=None):
    """
    Complete pipeline with choice of normalization
    
    Args:
        data_mesh: Input data dictionary
        output_type: 'drag', 'srp', or 'all'
        batch_size: Batch size for DataLoaders
        seed: Random seed
        normalization: 'minmax' ([-1,1]), 'maxabs' (scale by max|x|), 'zscore' (standard), 
                      'quantile' (uniform distribution), or 'none'
    """
    # 1. Convert to DataFrame
    df = prepare_dataframe(data_mesh)
    
    # 2. Non-dimensionalize
    df = factor_targets(df, data_mesh['sim_data'])
    print("-----------------------------------",output_type )
    # 3. Select outputs
    if output_type == 'drag_f':
        col_out = COL_OUT_DRAG_F
    elif output_type == 'drag_t':
        col_out = COL_OUT_DRAG_T
    elif output_type == "drag":
        col_out = COL_OUT_DRAG
    elif output_type == 'srp_f':
        col_out = COL_OUT_SRP_F
    elif output_type == 'srp_t':
        col_out = COL_OUT_SRP_T
    elif output_type == 'srp':
        col_out = COL_OUT_SRP
    elif output_type == 'all':
        col_out = COL_OUT_ALL
    else:
        raise ValueError("output_type must be 'drag', 'srp', or 'all'")
    
    # 4. Split FIRST
    train_df, tmp = train_test_split(df, test_size=0.30, random_state=seed)
    val_df, test_df = train_test_split(tmp, test_size=0.5, random_state=seed)

    if threshold is not None:
        # Get original size
        original_train_size = len(train_df)

        # Get all columns to check (inputs + outputs)
        all_cols = COL_IN + col_out

        # Create mask: keep samples where ALL absolute values are >= threshold
        mask = (train_df[all_cols].abs() >= threshold).all(axis=1)

        # Filter training set
        train_df = train_df[mask].reset_index(drop=True)

        # Report filtering results
        removed = original_train_size - len(train_df)
        print(f"\n=== Threshold Filtering (training set only) ===")
        print(f"Threshold: {threshold}")
        print(f"Original training samples: {original_train_size}")
        print(f"Samples removed: {removed} ({100 * removed / original_train_size:.1f}%)")
        print(f"Remaining training samples: {len(train_df)}")
        print(f"Validation samples (unchanged): {len(val_df)}")
        print(f"Test samples (unchanged): {len(test_df)}")
        print("=" * 50)

    # 5. Normalize using training stats only
    if normalization == 'minmax':
        train_df, val_df, test_df, scaler = minmax_scale_outputs(
            train_df, val_df, test_df, col_out
        )
        print(f"Applied Min-Max normalization to [-1, 1]")
        
    elif normalization == 'maxabs':
        train_df, val_df, test_df, scaler = maxabs_scale_outputs(
            train_df, val_df, test_df, col_out
        )
        print(f"Applied Max-Abs normalization (divide by max|x|)")
        
    elif normalization == 'zscore':
        train_df, val_df, test_df, scaler = zscale_outputs(
            train_df, val_df, test_df, col_out
        )
        print(f"Applied Z-score normalization")
        
    elif normalization == 'quantile':
        train_df, val_df, test_df, scaler = quantile_scale_outputs(
            train_df, val_df, test_df, col_out
        )
        print(f"Applied Quantile normalization (uniform distribution)")
        
    elif normalization == "none":
        scaler = None
    else:
        raise ValueError("normalization must be 'minmax', 'maxabs', 'zscore', 'quantile', or 'none'")

    
    # Print normalization statistics
    print(f"\nNormalized output ranges:")
    for col in col_out:
        print(f"  {col}: [{train_df[col].min():.3f}, {train_df[col].max():.3f}]")
    
    # 6. Create loaders
    train_loader, val_loader, test_loader, vX, vY, tX, tY = make_loaders(
        train_df, val_df, test_df, COL_IN, col_out, batch_size
    )
    
    return [train_loader, val_loader, test_loader, vX, vY, tX, tY, scaler, col_out]



# ==========================
# Model / train / eval
# ==========================
class MLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, hidden=16, layers=2, activation="relu"):
        super().__init__()
        acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "sigmoid": nn.Sigmoid}
        Act = acts[activation]
        seq = []
        last = in_dim
        for _ in range(layers):
            seq += [nn.Linear(last, hidden), Act()]
            last = hidden
        seq += [nn.Linear(last, out_dim)]  # bounded head like your baseline
        self.net = nn.Sequential(*seq)
    
    def forward(self, x): 
        return self.net(x)

def run_train(model, train_loader, val_loader, epochs, lr, weight_decay=0.0, patience=20, device=DEVICE):
    model.to(device)
    crit = nn.L1Loss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = np.inf
    best_state = None
    no_improve = 0
    
    for ep in tqdm(range(epochs), desc="Training"):
        model.train()
        train_loss = 0.0
        train_n = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
            train_n += xb.size(0)
        
        # Validation
        model.eval()
        vtot = 0.0
        vn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                vtot += crit(model(xb), yb).item() * xb.size(0)
                vn += xb.size(0)
        
        val_loss = vtot / max(vn, 1)
        
        # Early stopping
        if val_loss < best - 1e-9:
            best = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {ep+1}")
                break
        
        # Print progress every 10 epochs
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1}: Train Loss = {train_loss/train_n:.6f}, Val Loss = {val_loss:.6f}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def run_train_batch(model, train_loader, val_loader, epochs, lr, weight_decay=0.0, patience=20, device=DEVICE):
    """
    Enhanced training function with batch processing and monitoring
    """
    model.to(device)
    crit = nn.L1Loss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # For tracking
    train_losses = []
    val_losses = []
    best = np.inf
    best_state = None
    no_improve = 0
    best_epoch = 0
    
    print(f"Training on {len(train_loader.dataset)} samples with batch size {train_loader.batch_size}")
    print(f"Validation on {len(val_loader.dataset)} samples")
    print(f"Total batches per epoch: {len(train_loader)}")
    
    for ep in tqdm(range(epochs), desc="Training"):
        # ==================== TRAINING ====================
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)
            
            # Forward pass
            pred = model(xb)
            loss = crit(pred, yb)
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # Accumulate loss
            train_loss += loss.item() * xb.size(0)
            train_batches += 1
        
        # Average training loss for this epoch
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # ==================== VALIDATION ====================
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_loss += crit(pred, yb).item() * xb.size(0)
        
        # Average validation loss for this epoch
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # ==================== EARLY STOPPING ====================
        if avg_val_loss < best - 1e-9:
            best = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            best_epoch = ep + 1
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {ep+1}")
                print(f"Best validation loss: {best:.6f} at epoch {best_epoch}")
                break
        
        # ==================== LOGGING ====================
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"\nEpoch {ep+1}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss:   {avg_val_loss:.6f}")
            print(f"  Best Val:   {best:.6f} (epoch {best_epoch})")
            print(f"  No improve: {no_improve}/{patience}")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model from epoch {best_epoch}")
    
    return model, best, train_losses, val_losses


def evaluate_unscale(model, X, Y, means, stds, device=DEVICE):
    model.eval()
    X = X.to(device)
    Y = Y.to(device)
    
    with torch.no_grad():
        P = model(X)
    
    means_t = torch.tensor(means, dtype=P.dtype, device=P.device).view(1, -1)
    stds_t = torch.tensor(stds, dtype=P.dtype, device=P.device).view(1, -1)
    P_real = P * stds_t + means_t
    Y_real = Y * stds_t + means_t
    
    mae = torch.mean(torch.abs(P_real - Y_real)).item()
    rmse = torch.sqrt(torch.mean((P_real - Y_real)**2)).item()
    
    return mae, rmse


def get_prediction_scaled(model, X, Y, scaler, device=DEVICE):
    model.eval()
    X = X.to(device)
    Y = Y.to(device)
    
    with torch.no_grad():
        P = model(X)
    
    P_np = P.cpu().numpy()
    Y_np = Y.cpu().numpy()
    X_np = X.cpu().numpy()
    return P_np, Y_np, X_np


def get_predictions_unscaled(model, X, Y, scaler, normalization='quantile', device=DEVICE):
    """
    Get unscaled predictions and ground truth.
    
    Args:
        model: Trained model
        X: Input tensor
        Y: Target tensor (normalized)
        scaler: Scaler object (MinMaxScaler, dict of QuantileTransformers, etc.)
        normalization: Type of normalization used ('minmax', 'maxabs', 'zscore', 'quantile')
        device: Device to run on
    
    Returns: P_real, Y_real, X_real as numpy arrays (in original scale)
    """
    
    P_np, Y_np, X_np = get_prediction_scaled(model, X, Y, scaler, device=device)
    
    if scaler is None:
        return P_np, Y_np, X_np
    
    # Handle quantile scaler (dict of transformers)
    if isinstance(scaler, dict):
        P_real = np.zeros_like(P_np)
        Y_real = np.zeros_like(Y_np)
        
        col_names = list(scaler.keys())  # ['Fx', 'Fy', 'Fz'] or similar
        
        for i, col in enumerate(col_names):
            P_real[:, i:i+1] = scaler[col].inverse_transform(P_np[:, i:i+1])
            Y_real[:, i:i+1] = scaler[col].inverse_transform(Y_np[:, i:i+1])
        
        return P_real, Y_real, X_np
    
    # Handle standard scalers (MinMaxScaler, etc.)
    else:
        P_real = scaler.inverse_transform(P_np)
        Y_real = scaler.inverse_transform(Y_np)
        
        return P_real, Y_real, X_np


def load_ann_models(model_paths, device='cpu'):
    """
    Load multiple ANN models for different force/torque components

    Parameters:
    -----------
    model_paths : dict
        Dictionary with keys: 'drag_f', 'drag_t', 'srp_f', 'srp_t'
        Each value is the path to the corresponding model pickle file
    device : str
        'cpu' or 'cuda'

    Returns:
    --------
    models : dict
        Dictionary containing loaded models and scalers:
        {
            'drag_f': {'model': model, 'scaler': scaler},
            'drag_t': {'model': model, 'scaler': scaler},
            'srp_f': {'model': model, 'scaler': scaler},
            'srp_t': {'model': model, 'scaler': scaler}
        }

    Example:
    --------
    model_paths = {
        'drag_f': 'path/to/model_drag_f.pkl',
        'drag_t': 'path/to/model_drag_t.pkl',
        'srp_f': 'path/to/model_srp_f.pkl',
        'srp_t': 'path/to/model_srp_t.pkl'
    }
    models = load_ann_models(model_paths)
    """
    models = {}

    for model_type, path in model_paths.items():
        if path is None:
            models[model_type] = None
            continue

        try:
            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)

            # Check if this is a checkpoint format (with model_state_dict)
            if 'model_state_dict' in checkpoint and 'model_architecture' in checkpoint:
                # Checkpoint format
                arch = checkpoint['model_architecture']
                activation = arch['activation']
                hidden = arch['hidden']
                layers = arch['layers']

                model = MLP(in_dim=3, out_dim=3, hidden=layers, activation=activation)
                model.load_state_dict(checkpoint['model_state_dict'])
                scaler = checkpoint['scaler']

                print(f"  Loaded {model_type}: {path}")
                print(f"    Architecture: {layers} layers × {hidden} neurons, activation={activation}")
            else:
                # Old format (direct model object)
                model = checkpoint['model']
                scaler = checkpoint['scaler']
                print(f"  Loaded {model_type}: {path}")

            model.to(device)
            model.eval()

            models[model_type] = {
                'model': model,
                'scaler': scaler
            }

        except Exception as e:
            print(f"  Warning: Could not load {model_type} from {path}")
            print(f"    Error: {e}")
            models[model_type] = None

    return models


def load_ann_model(model_path, device='cpu'):
    """
    Load single ANN model from checkpoint

    Parameters:
    -----------
    model_path : str
        Path to model pickle file
    device : str
        'cpu' or 'cuda'

    Returns:
    --------
    model : torch.nn.Module
        Loaded model
    scaler : MinMaxScaler
        Input normalization scaler
    """
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)

    # Check if this is a checkpoint format
    if 'model_state_dict' in checkpoint and 'model_architecture' in checkpoint:
        # Checkpoint format
        arch = checkpoint['model_architecture']
        activation = arch['activation']
        hidden = arch['hidden']
        layers = arch['layers']

        model = MLP(in_dim=3, out_dim=3, hidden=hidden, layers=layers, activation=activation)
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = checkpoint['scaler']
    else:
        # Old format
        model = checkpoint['model']
        scaler = checkpoint['scaler']

    model.to(device)
    model.eval()
    return     {
        'model': model,
        'scaler': scaler
    }


def ann_predict_force(input_body, model_dict, device='cpu'):
    """
    Predict force using ANN model

    Parameters:
    -----------
    input_body : np.ndarray (3,)
        Input vector in body frame
    model_dict : dict
        Dictionary with 'model' and 'scaler' keys
    device : str
        'cpu' or 'cuda'

    Returns:
    --------
    F_pred : np.ndarray (3,)
        Predicted force [N]
    """
    if model_dict is None:
        return np.zeros(3)

    model = model_dict['model']
    scaler = model_dict['scaler']

    # Input: velocity (no normalization on input)
    input_body /= np.linalg.norm(input_body)
    v_tensor = torch.tensor(input_body.reshape(1, -1), dtype=torch.float32).to(device)

    # Predict (output is normalized)
    with torch.no_grad():
        F_pred_norm = model(v_tensor).cpu().numpy()

    # Unscale output
    if scaler is not None:
        F_pred = scaler.inverse_transform(F_pred_norm).flatten()
    else:
        F_pred = F_pred_norm.flatten()

    return F_pred


def ann_predict_torque(input_body, model_dict, device='cpu'):
    """
    Predict torque using ANN model

    Parameters:
    -----------
    input_body : np.ndarray (3,)
        Input vector in body frame
    model_dict : dict
        Dictionary with 'model' and 'scaler' keys
    device : str
        'cpu' or 'cuda'

    Returns:
    --------
    T_pred : np.ndarray (3,)
        Predicted torque [N⋅m]
    """
    if model_dict is None:
        return np.zeros(3)

    model = model_dict['model']
    scaler = model_dict['scaler']

    # Input: velocity (no normalization on input)
    input_body /= np.linalg.norm(input_body)
    v_tensor = torch.tensor(input_body.reshape(1, -1), dtype=torch.float32).to(device)

    # Predict (output is normalized)
    with torch.no_grad():
        T_pred_norm = model(v_tensor).cpu().numpy()

    # Unscale output
    if scaler is not None:
        T_pred = scaler.inverse_transform(T_pred_norm).flatten()
    else:
        T_pred = T_pred_norm.flatten()

    return T_pred


def ann_predict(velocity_body, model, scaler, device='cpu'):
    """
    Predict force using ANN model (legacy single model support)

    Parameters:
    -----------
    velocity_body : np.ndarray (3,)
        Velocity vector in body frame [m/s]
    model : torch.nn.Module
        Trained neural network
    scaler : MinMaxScaler
        Input normalization scaler

    Returns:
    --------
    F_pred : np.ndarray (3,)
        Predicted force [N]
    """
    # Normalize input
    v_normalized = scaler.transform(velocity_body.reshape(1, -1))

    # Convert to tensor
    v_tensor = torch.tensor(v_normalized, dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        F_pred = model(v_tensor).cpu().numpy().flatten()

    return F_pred
