# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 09:55:10 2025

@author: lushi
"""

from __future__ import annotations

import time

import pyvista as pv
import numpy as np
import vtk
from tqdm import tqdm
from scipy.special import erf
import matplotlib.pyplot as plt
from monitor import show_ray_tracing, show_ray_tracing_fast
from ray_tracing import compute_ray_tracing, compute_ray_tracing_fast
# a spherical model of ray tracing to validate that our model and force calculations are accurate

mesh = pv.Sphere(radius = 100)
mesh = mesh.triangulate().clean()
#mesh.plot(show_edges=True)

ray_direction = np.array([0,0, -1])
ray_direction = ray_direction / np.linalg.norm(ray_direction) #normalise

bounds = mesh.bounds #(xmin, xmax, ymin, ymax, zmin, zmax)

resolutions = [50, 100, 200]
A_ = np.pi * (0.1)**2
area_errors = []

#loop over resolutions to find the optimum 
#optimum is where our total estimated area on pixel array (len(hit_points)*pixel_area) = pi*r^2 with 1% error

for res in tqdm(resolutions, desc="Resolution"):
    res_x = res
    res_y = res

    #z_start = bounds[5] + 20
    #t0 = time.time()
    #res_prop = compute_ray_tracing(mesh, res_x, res_y)
    #print("time 1:", time.time() - t0)

    t0 = time.time()
    res_prop = compute_ray_tracing_fast(mesh, res_x, res_y)
    print("time 2:", time.time() - t0)

    # show_ray_tracing(mesh, res_prop)
    show_ray_tracing_fast(mesh, res_prop)

    hit_points = res_prop['hit_points']

    pixel_width = res_prop['pixel_width']
    pixel_height = res_prop['pixel_height']
    pixel_area = pixel_width*pixel_height

    area = len(hit_points)*pixel_area
    area_error = (area - A_)/A_
    
    area_errors.append(area_error)
    print(area_error)
    if abs(area_error) <= 0.01:
        break

    #now i have coords where the rays hit  
    #ran res=2000 to find area_error = 0.01123518784316649 but unable to run higher res on my laptop
    
#%%
#graph for area error to resolution loop


plt.figure(figsize=(6,4)) 
plt.plot(resolutions, area_errors, color = 'blue', linewidth = 1)
plt.xlabel("Resolutions")
plt.ylabel("Area Error")
    
plt.grid(True)

plt.show()
#%%
# finding illuminated cells 

vtk_mesh = mesh.GetDataSet() if hasattr(mesh, "GetDataSet") else mesh #convert mesh to vtk

# Create and build vtkcell locator
locator = vtk.vtkCellLocator()
locator.SetDataSet(vtk_mesh)
locator.BuildLocator()

#unique illuminated cells
#illuminated_cells = set()
#for pt in hit_points:
#    cid = locator.FindCell(pt)
#    if cid != -1: #not null
#        illuminated_cells.add(cid)
        
#illuminated_cells = np.array(list(illuminated_cells))

#non unique illuminated cells
cell_ids_per_ray = []
hit_points_per_ray = []

for pt in hit_points:
    cid = locator.FindCell(pt)
    if cid != -1:
        cell_ids_per_ray.append(cid)
        hit_points_per_ray.append(pt)

#%%
#normal_vectors = mesh.cell_normals[illuminated_cells]  # array of normals, repeated cells included
normal_vectors = mesh.cell_normals[cell_ids_per_ray] 
 
norms = np.linalg.norm(normal_vectors, axis=1, keepdims=True)
normal_vectors = -1*(normal_vectors / norms)  #normalise

ref_vectors = np.cross(ray_direction, normal_vectors) 
rf_norms = np.linalg.norm(ref_vectors, axis=1, keepdims=True)
safe_norms = np.where(rf_norms > 0, rf_norms, 1) #if norm > 0 use actual norm, if not then use 1
ref_vectors = ref_vectors / safe_norms
ref_vectors[norms[:, 0] <= 0] = 0 #if norm is <=0 then it sets vector to 0

tangent_vectors = np.cross(-ref_vectors, normal_vectors)

t_norms = np.linalg.norm(tangent_vectors, axis = 1, keepdims=True)
safe_t_norms = np.where(t_norms > 0, t_norms, 1)
tangent_vectors = tangent_vectors / safe_t_norms
tangent_vectors[norms[:, 0] <= 0] = 0

#%%
#com vectors
com = mesh.center
com = np.array(com)
com_vectors = hit_points-com 

#cell_centers = mesh.cell_centers().points
#com_vectors = cell_centers[illuminated_cells] - com

#%%
#now find projected area of each pixel

cos_theta = np.dot(normal_vectors, ray_direction)

mask = np.abs(cos_theta) > 1e-6 #filter out very small cos to avoid blow ups
                                #apply mask to all defs to ensure the same size
cos_theta = cos_theta[mask] 
normal_vectors = normal_vectors[mask]     
tangent_vectors = tangent_vectors[mask]   
com_vectors = com_vectors[mask]            

proj_areas = pixel_area/cos_theta 

#computing actual area of each cell (alternative method)

#mesh_with_sizes = mesh.compute_cell_sizes()
#all_cell_areas = mesh_with_sizes.cell_data['Area'] 
#original_cell_area = all_cell_areas[illuminated_cells] *0.001 * 0.001  

#multiply area by cos_theta for projected area
#proj_areas = original_cell_area * cos_theta  

    #%%
Temp_molecular = 2
C2K = 273.15
spec_ = 0.8
T_wall = 30 + C2K # wall temperature
v_mag = 7800 # [m/s] velocity mag
MM  = 18 #e-3 / 6.022e23?  #18 kg/m
#18.0 # molecular weight of air in space [kg/mol]
kB = 1.3806488e-23 # [J/K]Boltzman Coeff
diffuse = 0.2

def calc_cn_ct_numba(cos_theta, sin_theta, s_, spec_, T_wall):
    """Calcula Cn y Ct para un panel plano"""
    Sn = s_ * cos_theta
    St = s_ * sin_theta
    term1 = ((2.0 - diffuse) / np.sqrt(np.pi)) * func_Pi_numba(Sn) / (s_ ** 2)
    term2 = (diffuse / 2.0) * func_Chi_numba(Sn) / (s_ ** 2) * np.sqrt(T_wall/ (Temp_molecular + C2K))  # match with original
    cn = term1 + term2
    ct = diffuse * St * func_Chi_numba(Sn) / (np.sqrt(np.pi) * s_ ** 2)
    return cn, ct

def func_Pi_numba(s):
    return s * np.exp(-s ** 2) + np.sqrt(np.pi) * (s ** 2 + 0.5) * (1 + erf(s))


def func_Chi_numba(s):
    return np.exp(-s ** 2) + np.sqrt(np.pi) * s * (1 + erf(s))

def calc_force_d(q_vel, cn, ct, n_, t_, area_):
    return q_vel * area_ * (cn * n_ + ct * t_) 

def calc_torque_d(c_, force_d):
    return np.cross(c_ , force_d)
    
def calc_force_s(P, area_, cos_theta, diffuse, ray_direction, spec_, n_):
    return P*area_*cos_theta*(diffuse*ray_direction + 2*(spec_*cos_theta + (1/3)*diffuse)*n_)
    
def calc_torque_s(c_, force_s):
    return np.cross(c_, force_s)
        
forces_d = []
torques_d = []
forces_s = []
torques_s = []
den = 5e-13
q_vel = 0.5 * (v_mag **2)* den

for n_ , t_ , c_, area_ in zip(normal_vectors, tangent_vectors, com_vectors, proj_areas):
        
    cos_theta = n_ @ ray_direction
    sin_theta = np.sin(np.arccos(cos_theta))
    P = 4.57e-6

    vm = np.sqrt(2.0 * kB * T_wall / MM)
    s_ = v_mag / vm

    cn, ct = calc_cn_ct_numba(cos_theta, sin_theta, s_, spec_, T_wall)
        
    force_d = calc_force_d(q_vel, cn, ct, n_, t_, area_)
    forces_d.append(force_d)
        
    torque_d = calc_torque_d(c_, force_d)
    torques_d.append(torque_d)
        
    force_s = calc_force_s(P, area_, cos_theta, diffuse, ray_direction, spec_, n_)
    forces_s.append(force_s)
        
    torque_s = calc_torque_s(c_, force_s)
    torques_s.append(torque_s)
        
F_d = sum(forces_d)   
T_d = sum(torques_d)   
F_s = sum(forces_s)   
T_s = sum(torques_s)



    
#%%
#real drag force

A = 2*np.pi * (0.1)**2 #radius = 0.1m # area is half surface area of sphere
v = ray_direction*v_mag
F_R = 0.5 * den * v_mag * v * A

print(F_R, F_d)

error = np.linalg.norm(F_d - F_R)/np.linalg.norm(F_R)

plt.show()