# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 13:47:45 2025

@author: lushi
"""

# feed forward deep neural network

# to work in future use a virtual environment 

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
import time
import ray
import optuna

v_mag = 7800
den = 5e-13
P = 4.57e-6

ta = time.time()

# sorting the data

df_original = pd.read_csv('data_1751144053.6894915.csv')
df = pd.read_csv('data_1751144053.6894915.csv')
df['f_x_d'] = df['f_x_d'].multiply(2* 1/den * 1/(v_mag**2)) # factoring out 0.5*den*v^2 as we dont want these parameters in the neural network
df['f_y_d'] = df['f_y_d'].multiply(2* 1/den * 1/(v_mag**2))
df['f_z_d'] = df['f_z_d'].multiply(2* 1/den * 1/(v_mag**2))

df['t_x_d'] = df['t_x_d'].multiply(2* 1/den * 1/(v_mag**2)) 
df['t_y_d'] = df['t_y_d'].multiply(2* 1/den * 1/(v_mag**2))
df['t_z_d'] = df['t_z_d'].multiply(2* 1/den * 1/(v_mag**2))

#take out P for the srp calcs
df['f_x_s'] = df['f_x_s'].multiply(1/P)
df['f_y_s'] = df['f_y_s'].multiply(1/P)
df['f_z_s'] = df['f_z_s'].multiply(1/P)

df['t_x_s'] = df['t_x_s'].multiply(1/P)
df['t_y_s'] = df['t_y_s'].multiply(1/P)
df['t_z_s'] = df['t_z_s'].multiply(1/P)

raw_force = df['f_x_d'].values
raw_torque = df['t_x_d'].values


means = torch.tensor([df['f_x_d'].mean(),df['f_y_d'].mean(),df['f_z_d'].mean(),df['t_x_d'].mean(),
                      df['t_y_d'].mean(),df['t_z_d'].mean(),df['f_x_s'].mean(),df['f_y_s'].mean(),
                      df['f_z_s'].mean(),df['t_x_s'].mean(),df['t_y_s'].mean(),df['t_z_s'].mean()])

stds = torch.tensor([df['f_x_d'].std(),df['f_y_d'].std(),df['f_z_d'].std(),df['t_x_d'].std(),
                      df['t_y_d'].std(),df['t_z_d'].std(),df['f_x_s'].std(),df['f_y_s'].std(),
                      df['f_z_s'].std(),df['t_x_s'].std(),df['t_y_s'].std(),df['t_z_s'].std()])

means = means.numpy()
stds = stds.numpy()

for col in df.columns:
    if col not in ['v_x', 'v_y', 'v_z']:
        df[col] = ((df[col] - df[col].mean()) / df[col].std())*0.1 # z-normalising data



from sklearn.model_selection import train_test_split # splitting data up into train, val, test

train_df, val_df = train_test_split(df, test_size=0.3, random_state=42) 
val_df, test_df = train_test_split(val_df, test_size=1/3, random_state=42) 

X = torch.tensor(train_df[['v_x', 'v_y', 'v_z']].values, dtype=torch.float32) #inputs
y = torch.tensor(train_df[['f_x_d', 'f_y_d', 'f_z_d', 't_x_d', 't_y_d', 
                           't_z_d', 'f_x_s', 'f_y_s', 'f_z_s', 't_x_s',
                           't_y_s', 't_z_s']].values, dtype=torch.float32)

val_X = torch.tensor(val_df[['v_x', 'v_y', 'v_z']].values, dtype=torch.float32)
val_y = torch.tensor(val_df[['f_x_d', 'f_y_d', 'f_z_d', 't_x_d', 't_y_d', 
                           't_z_d', 'f_x_s', 'f_y_s', 'f_z_s', 't_x_s',
                           't_y_s', 't_z_s' ]].values, dtype=torch.float32)

test_X = torch.tensor(test_df[['v_x', 'v_y', 'v_z']].values, dtype=torch.float32)
test_y = torch.tensor(test_df[['f_x_d', 'f_y_d', 'f_z_d', 't_x_d', 't_y_d', 
                           't_z_d', 'f_x_s', 'f_y_s', 'f_z_s', 't_x_s',
                           't_y_s', 't_z_s' ]].values, dtype=torch.float32)



#batch the data
train_dataset = TensorDataset(X, y) 
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

#%%
# neural network and validation

class neural_network(nn.Module):
    def __init__(self):
        super(neural_network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 20),   # input: 3 features 20 hidden neurons
            nn.ReLU(),          # activation func adds non linearity
            nn.Linear(20, 20),  
            nn.ReLU(),          
            nn.Linear(20, 12),
            nn.Tanh()     # output: 12
        )
    def forward(self, x):
        return self.model(x) # defines what happens so for x, we apply it to the nn
    
    
model = neural_network()
criterion = nn.MSELoss()                # mean squared error loss # loss func # good for cts values
optimizer = optim.Adam(model.parameters(), lr=0.001) # updates weights in model based on gradients. lr is learning rate # our optimisation

    
#training and validating

train_losses = []
val_losses = []

for epoch in range(80):
    epoch_loss = 0
    model.train()
    
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_X.size(0)  # as total epoch loss for that batch = average loss * no of data
    epoch_loss /= len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    model.eval()
    with torch.no_grad():  # no gradient tracking
        val_outputs = model(val_X)
        val_loss = criterion(val_outputs, val_y)
        val_losses.append(val_loss.item())
        
        if val_loss>epoch_loss:
            break
        
tb = time.time()
print("Elapsed time:", tb - ta, "seconds")

#%%
# optimising using optuna


def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)
    hidden_size = trial.suggest_categorical("hidden_size", [20, 30, 35, 40])
    batch_size = trial.suggest_categorical("batch_size", [20, 35, 50, 70])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    class neural_network(nn.Module):
        def __init__(self):
            super(neural_network, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 12),
                nn.Tanh()
            )

        def forward(self, x):
            return self.model(x)

    model = neural_network()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(90):  
        epoch_loss = 0
        model.train()

        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X)
            val_loss = criterion(val_outputs, val_y)
            val_losses.append(val_loss.item())

    return min(val_losses)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Print best result
print("Best params:", study.best_params)
#ta = time.time()
#tb = time.time()

#%%
#graph to show validation and training

plt.figure(figsize=(6,4)) 
plt.yscale('log')
plt.plot(train_losses, label="Train Loss", color = 'blue', linewidth = 1)
plt.plot(val_losses, label="Validation Loss", color = 'red', linewidth = 1)
plt.xlabel("Epoch")
plt.ylabel("Loss")
    
plt.grid(True)
plt.show()

#now test 
model.eval()  
real_outputs = test_y

with torch.no_grad(): 
    pred_outputs = model(test_X) #predicted outputs of test inputs using nn

#real outputs from test data
#reversing the z normalisation we did

cols = [ 'f_x_d', 'f_y_d', 'f_z_d', 't_x_d', 't_y_d', 't_z_d',
        'f_x_s', 'f_y_s', 'f_z_s', 't_x_s', 't_y_s', 't_z_s']

real_test_df = test_df.copy()
for col, mean, std in zip(cols, means, stds):
    real_test_df[col] = real_test_df[col] * std * 10 + mean


real_f_x_d = real_test_df['f_x_d']
real_f_y_d = real_test_df['f_y_d']
real_f_z_d = real_test_df['f_z_d']
real_t_x_d = real_test_df['t_x_d']
real_t_y_d = real_test_df['t_y_d']
real_t_z_d = real_test_df['t_z_d']
real_f_x_s = real_test_df['f_x_s']
real_f_y_s = real_test_df['f_y_s']
real_f_z_s = real_test_df['f_z_s']
real_t_x_s = real_test_df['t_x_s']
real_t_y_s = real_test_df['t_y_s']
real_t_z_s = real_test_df['t_z_s']

means = torch.tensor(means, dtype=pred_outputs.dtype, device=pred_outputs.device).view(1, -1)
stds = torch.tensor(stds, dtype=pred_outputs.dtype, device=pred_outputs.device).view(1, -1)

pred_outputs = pred_outputs * stds * 10 + means

pred_f_x_d = pred_outputs[:,0]
pred_f_y_d = pred_outputs[:,1]
pred_f_z_d = pred_outputs[:,2]
pred_t_x_d = pred_outputs[:,3]
pred_t_y_d = pred_outputs[:,4]
pred_t_z_d = pred_outputs[:,5]
pred_f_x_s = pred_outputs[:,6]
pred_f_y_s = pred_outputs[:,7]
pred_f_z_s = pred_outputs[:,8]
pred_t_x_s = pred_outputs[:,9]
pred_t_y_s = pred_outputs[:,10]
pred_t_z_s = pred_outputs[:,11]
#%%

import matplotlib.pyplot as plt
from scipy.stats import linregress

# test data plotted with nn estimates with r^2 annotation
def scatter_with_r2(
    ax, 
    x, 
    y, 
    xlabel, 
    ylabel, 
    color='blue', 
    alpha=0.3, 
    s=5, 
    xlim=None, 
    ylim=None
):
    ax.scatter(x, y, color=color, alpha=alpha, s=s)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
    #compute R^2
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value**2
    
    ax.text(
        0.05, 0.90, 
        f"$R^2$ = {r_squared:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
    )

fig, axs = plt.subplots(4, 3, figsize=(12, 9))

# Row 0
scatter_with_r2(
    axs[0,0], real_f_x_d, pred_f_x_d, "Real f_x_d", "Pred f_x_d",
    xlim=(-0.015,0.015), ylim=(-0.015,0.015)
)
scatter_with_r2(
    axs[0,1], real_f_y_d, pred_f_y_d, "Real f_y_d", "Pred f_y_d",
    xlim=(-0.015,0.015), ylim=(-0.015,0.015)
)
scatter_with_r2(
    axs[0,2], real_f_z_d, pred_f_z_d, "Real f_z_d", "Pred f_z_d",
    xlim=(-0.025,0.025), ylim=(-0.025,0.025)
)

# Row 1
scatter_with_r2(
    axs[1,0], real_t_x_d, pred_t_x_d, "Real t_x_d", "Pred t_x_d"
)
scatter_with_r2(
    axs[1,1], real_t_y_d, pred_t_y_d, "Real t_y_d", "Pred t_y_d"
)
scatter_with_r2(
    axs[1,2], real_t_z_d, pred_t_z_d, "Real t_z_d", "Pred t_z_d",
    xlim=(-3,3), ylim=(-3,3)
)

# Row 2
scatter_with_r2(
    axs[2,0], real_f_x_s, pred_f_x_s, "Real f_x_s", "Pred f_x_s", color='purple'
)
scatter_with_r2(
    axs[2,1], real_f_y_s, pred_f_y_s, "Real f_y_s", "Pred f_y_s", color='purple',
    xlim=(-0.01,0.01), ylim=(-0.01,0.01)
)
scatter_with_r2(
    axs[2,2], real_f_z_s, pred_f_z_s, "Real f_z_s", "Pred f_z_s", color='purple'
)

# Row 3
scatter_with_r2(
    axs[3,0], real_t_x_s, pred_t_x_s, "Real t_x_s", "Pred t_x_s", color='purple'
)
scatter_with_r2(
    axs[3,1], real_t_y_s, pred_t_y_s, "Real t_y_s", "Pred t_y_s", color='purple',
    xlim=(-1.5,1.5), ylim=(-1.5,1.5)
)
scatter_with_r2(
    axs[3,2], real_t_z_s, pred_t_z_s, "Real t_z_s", "Pred t_z_s", color='purple'
)


plt.tight_layout()
plt.show()

#%%
# why do we z normalise? lets look at the data for F_d

plt.figure(figsize=(8,4))
plt.hist(raw_force, bins=50, color='blue', edgecolor='black')
plt.xlabel('Unnormalized force (f_x_d)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#%%
plt.figure(figsize=(8,4))
plt.hist(raw_torque, bins=50, color='blue', edgecolor='black')
plt.xlabel('Unnormalized torque (t_x_d)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#%%

plt.figure(figsize=(8,4))
plt.hist(df['f_x_d'].values, bins=50, color='blue', edgecolor='black')
plt.xlabel('Normalized f_x_d')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#%%
normalised_torque = df['t_x_d'].values

plt.figure(figsize=(8,4))
plt.hist(df['t_x_d'].values, bins=50, color='blue', edgecolor='black')
plt.xlabel('Normalised t_x_d')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



