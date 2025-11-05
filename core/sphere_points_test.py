# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 16:56:40 2025

@author: mndc5
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import qmc
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "serif",                     # generic family
    "font.serif": ["Times New Roman", "Times"], # try exact TNR first
    "font.size": 18,                            # default text size
    "axes.titlesize": 18,                       # axes title
    "axes.labelsize": 18,                       # x/y labels
    "xtick.labelsize": 18,                      # tick labels
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "figure.titlesize": 18,
    # Math text configured to look like Times
    "mathtext.fontset": "stix",                 # STIX resembles Times
    "mathtext.rm": "Times New Roman",
})



def rotation_matrix_from_vectors(a, b):
    """Find rotation matrix that rotates vector a to vector b"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    # Skew-symmetric cross-product matrix
    kmat = np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])
    
    # Rodrigues' formula
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2 + 1e-10))
    
    return R


def lhs_sphere(n_samples, seed=None):
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    sample = sampler.random(n=n_samples)
    
    u, v = sample[:, 0], sample[:, 1]
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    
    x = -np.sin(phi) * np.cos(theta)
    y = -np.sin(phi) * np.sin(theta)
    z = -np.cos(phi)
    
    return np.column_stack([x, y, z])

def halton_sequence(n, base):
    sequence = []
    for i in range(n):
        f, r = 1.0, 0.0
        ii = i
        while ii > 0:
            f = f / base
            r = r + f * (ii % base)
            ii = ii // base
        sequence.append(r)
    return np.array(sequence)

def halton_sphere_old(n_samples, seed=None):
    u = halton_sequence(n_samples, base=2)
    v = halton_sequence(n_samples, base=3)
    
    if seed is not None:
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        u, v = u[indices], v[indices]
    
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    
    x = -np.sin(phi) * np.cos(theta)
    y = -np.sin(phi) * np.sin(theta)
    z = -np.cos(phi)
    
    return np.column_stack([x, y, z])


def halton_sphere(n_samples, seed=None):
    """
    Generate uniformly distributed points using Halton sequence from scipy.
    """
    sampler = qmc.Halton(d=2, scramble=True, seed=seed)
    sample = sampler.random(n=n_samples)
    
    u, v = sample[:, 0], sample[:, 1]
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    
    x = -np.sin(phi) * np.cos(theta)
    y = -np.sin(phi) * np.sin(theta)
    z = -np.cos(phi)
    
    return np.column_stack([x, y, z])

def fibonacci_sphere(n_samples):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))
    
    for i in range(n_samples):
        y = 1 - (i / float(n_samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([-x, -y, -z])
    
    return np.array(points)

def compute_uniformity_metrics(vectors):
    nn_distances = []
    sample_size = len(vectors)
    sample_indices = np.random.choice(len(vectors), sample_size, replace=False)
    
    for i in sample_indices:
        other_points = np.delete(vectors, i, axis=0)
        dists = np.linalg.norm(other_points - vectors[i], axis=1)
        nn_distances.append(np.min(dists))
    
    nn_distances = np.array(nn_distances)
    cv = nn_distances.std() / nn_distances.mean()
    
    # Map sphere vectors back to [0,1]^2 for discrepancy calculation
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    theta = np.arctan2(y, x)
    phi = np.arccos(-z)
    u = theta / (2 * np.pi)
    v = (np.cos(phi) + 1) / 2
    
    # Normalize to [0,1]
    u = (u + 0.5) % 1.0
    v = np.clip(v, 0, 1)
    
    uv_samples = np.column_stack([u, v])
    disc = qmc.discrepancy(uv_samples)

    return {
        'mean_nn': nn_distances.mean(),
        'std_nn': nn_distances.std(),
        'cv': cv,
        'min_nn': nn_distances.min(),
        'max_nn': nn_distances.max(),
        'discrepancy': disc * 1000,
        'nn_distances': nn_distances
    }

def compare_all_methods(n_points=200):    
    lhs_vectors = lhs_sphere(n_points, seed=42)
    halton_vectors = halton_sphere(n_points, seed=42)
    fibonacci_vectors = fibonacci_sphere(n_points)
    print(len(lhs_vectors))
    lhs_metrics = compute_uniformity_metrics(lhs_vectors)
    halton_metrics = compute_uniformity_metrics(halton_vectors)
    fibonacci_metrics = compute_uniformity_metrics(fibonacci_vectors)
    
    fig = plt.figure(figsize=(16, 8))
    
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    ax1.scatter(lhs_vectors[:, 0], lhs_vectors[:, 1], lhs_vectors[:, 2],
               c='#2e7d32', s=30, alpha=0.8)
    ax1.set_title('LHS')
    ax1.set_box_aspect([1,1,1])
    ax1.view_init(elev=20, azim=45)
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    ax2.scatter(halton_vectors[:, 0], halton_vectors[:, 1], halton_vectors[:, 2],
               c='#1976d2', s=30, alpha=0.8)
    ax2.set_title('Halton')
    ax2.set_box_aspect([1,1,1])
    ax2.view_init(elev=20, azim=45)
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    ax3.scatter(fibonacci_vectors[:, 0], fibonacci_vectors[:, 1], fibonacci_vectors[:, 2],
               c='#f57c00', s=30, alpha=0.8)
    ax3.set_title('Fibonacci')
    ax3.set_box_aspect([1,1,1])
    ax3.view_init(elev=20, azim=45)
    #ax2.set_axis_off()
    ax4 = fig.add_subplot(2, 4, 4)
    categories = ['Mean NN', 'Std NN', 'Min NN', 'Max NN']
    lhs_vals = [lhs_metrics['mean_nn'], lhs_metrics['std_nn'], 
                lhs_metrics['min_nn'], lhs_metrics['max_nn']]
    halton_vals = [halton_metrics['mean_nn'], halton_metrics['std_nn'],
                   halton_metrics['min_nn'], halton_metrics['max_nn']]
    fib_vals = [fibonacci_metrics['mean_nn'], fibonacci_metrics['std_nn'],
                fibonacci_metrics['min_nn'], fibonacci_metrics['max_nn']]
    
    x = np.arange(len(categories))
    w = 0.15
    
    ax4.bar(x - w, lhs_vals, w, label='LHS', color='#2e7d32', alpha=0.8)
    ax4.bar(x, halton_vals, w, label='Halton', color='#1976d2', alpha=0.8)
    ax4.bar(x + w, fib_vals, w, label='Fib.', color='#f57c00', alpha=0.8)
    
    ax4.set_ylabel('Distance')
    ax4.set_title('Distance Metrics')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=14, rotation=15)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.scatter(lhs_vectors[:, 0], lhs_vectors[:, 1], c='#2e7d32', s=35, alpha=0.7)
    ax5.set_aspect('equal')
    ax5.set_title('LHS - Top View')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax5.add_patch(circle)
    ax5.set_xlim(-1.15, 1.15)
    ax5.set_ylim(-1.15, 1.15)
    
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.scatter(halton_vectors[:, 0], halton_vectors[:, 1], c='#1976d2', s=35, alpha=0.7)
    ax6.set_aspect('equal')
    ax6.set_title('Halton - Top View')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax6.add_patch(circle)
    ax6.set_xlim(-1.15, 1.15)
    ax6.set_ylim(-1.15, 1.15)
    
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.scatter(fibonacci_vectors[:, 0], fibonacci_vectors[:, 1], c='#f57c00', s=35, alpha=0.7)
    ax7.set_aspect('equal')
    ax7.set_title('Fibonacci - Top View')
    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax7.add_patch(circle)
    ax7.set_xlim(-1.15, 1.15)
    ax7.set_ylim(-1.15, 1.15)
    
    """
    ax8 = fig.add_subplot(2, 4, 8)
    categories = ['CV']
    vals = [[lhs_metrics['cv']], [halton_metrics['cv']], [fibonacci_metrics['cv']]]
    x = np.arange(len(categories))
    w = 0.25
    
    ax8.bar(x - w, vals[0], w, label='LHS', color='#2e7d32', alpha=0.8)
    ax8.bar(x, vals[1], w, label='Halton', color='#1976d2', alpha=0.8)
    ax8.bar(x + w, vals[2], w, label='Fibonacci', color='#f57c00', alpha=0.8)
    
    ax8.set_ylabel('CV')
    ax8.set_title('Uniformity')
    ax8.set_xticks(x)
    ax8.set_xticklabels(categories)
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    
    for i, v in enumerate(vals):
        ax8.text(x[0] + (i-1)*w, v[0], f'{v[0]:.4f}', 
                ha='center', va='bottom', fontsize=14)
        """
    ax8 = fig.add_subplot(2, 4, 8)
    categories = ['Discrepancy']
    vals = [[lhs_metrics['discrepancy']], [halton_metrics['discrepancy']], 
            [fibonacci_metrics['discrepancy']]]
    x = np.arange(len(categories))
    w = 0.15
    
    ax8.bar(x - w, vals[0], w, label='LHS', color='#2e7d32', alpha=0.8)
    ax8.bar(x, vals[1], w, label='Halton', color='#1976d2', alpha=0.8)
    ax8.bar(x + w, vals[2], w, label='Fib.', color='#f57c00', alpha=0.8)
    
    ax8.set_ylabel(r'Discrepancy $[1\times10^{3}]$')
    ax8.set_title('Space-Filling Quality')
    ax8.set_xticks(x)
    ax8.set_xticklabels(categories)
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(vals):
        ax8.text(x[0] + (i-1)*w, v[0], f'{v[0]:.4f}', 
                ha='center', va='bottom', fontsize=14)
    #ax8.set_xlim(-0.65, 0.25)
    ax8.set_ylim(0, 1.7*np.max(vals))
    plt.tight_layout()
    fig.savefig(f"sample_options_{n_points}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return lhs_vectors, halton_vectors, fibonacci_vectors

if __name__ == "__main__":
    lhs, halton, fib = compare_all_methods(n_points=500)