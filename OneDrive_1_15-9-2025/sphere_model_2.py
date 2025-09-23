# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 09:55:10 2025

@author: lushi
"""

from __future__ import annotations
import pyvista as pv
import numpy as np
import vtk 
from scipy.special import erf

# a spherical model of ray tracing to validate that our model and force calculations are accurate

mesh = pv.Sphere(radius = 100)
mesh = mesh.triangulate().clean()
#mesh.plot(show_edges=True)

ray_direction = np.array([0,0, -1])
ray_direction = ray_direction / np.linalg.norm(ray_direction) #normalise

bounds = mesh.bounds #(xmin, xmax, ymin, ymax, zmin, zmax)

resolutions = [     ]
A_ = np.pi * (0.1)**2
area_errors = []

#loop over resolutions to find the optimum 
#optimum is where our total estimated area on pixel array (len(hit_points)*pixel_area) = pi*r^2 with 1% error

for res in resolutions:
    res_x = res
    res_y = res

    pixel_width = 220 / res_x /1000 # into m
    pixel_height = 220 / res_y/1000
    pixel_area = pixel_width*pixel_height
    z_start = bounds[5] + 10
            
    x_range = np.linspace(-110, 110, res_x) #range of res_x number of values between -110 and 110
    y_range = np.linspace(-110, 110, res_y)

    hit_points = []
    ray_starts = []
    ray_ends = []
    
    xx, yy = np.meshgrid(x_range + pixel_width/2, y_range + pixel_height/2)
    zz = np.full_like(xx, z_start)

    starts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    directions = np.tile(ray_direction, (starts.shape[0], 1))

    points, ray_ids, cell_ids = mesh.multi_ray_trace(starts, directions, first_point=True)
    
    area = len(points)*pixel_area
    area_error = (area - A_)/A_
    print(area_error)
    area_errors.append(area_error)
    
    if abs(area_error) <= 0.01:
        break

    #now i have coords where the rays hit  
    #ran res=2000 to find area_error = 0.01123518784316649 but unable to run higher res on my laptop
    #ran at res=5000 with error = -0.010628017283151082
    
#%%
#graph for area error to resolution loop
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4)) 
plt.plot(resolutions, area_errors, color = 'blue', linewidth = 1)
plt.xlabel("Resolutions")
plt.ylabel("Area Error")
    
plt.grid(True)
plt.show()


#%%
#normal_vectors = mesh.cell_normals[illuminated_cells]  # array of normals, repeated cells included
normal_vectors = mesh.cell_normals[cell_ids] 
 
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
com_vectors = points-com 

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

proj_area = pixel_area/cos_theta 
sin_theta = np.sin(np.arccos(cos_theta))

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
MM = 18.0 # molecular weight of air in space [kg/mol]
kB = 1.3806488e-23 # [J/K]Boltzman Coeff
diffuse = 1.0 - spec_

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
    return q_vel * proj_area[:, np.newaxis] * (cn[:, np.newaxis] * normal_vectors + ct[:, np.newaxis] * tangent_vectors) 

def calc_torque_d(c_, force_d):
    return np.cross(com_vectors , force_d)

def calc_force_s(P, area_, cos_theta, diffuse, ray_direction, spec_, n_):
    return P*proj_area[:, np.newaxis]*cos_theta[:, np.newaxis]*(diffuse*ray_direction + 2*(spec_*cos_theta[:, np.newaxis] + (1/3)*diffuse)*normal_vectors)

def calc_torque_s(c_, force_s):
    return np.cross(com_vectors, force_s)
    
#    forces_d = []
#    torques_d = []
 #   forces_s = []
 #   torques_s = []
den = 5e-13
q_vel = 0.5 * (v_mag **2)* den
P = 4.57e-6

vm = np.sqrt(2.0 * kB * T_wall / MM)
s_ = v_mag / vm

cn, ct = calc_cn_ct_numba(cos_theta, sin_theta, s_, spec_, T_wall)

force_d = calc_force_d(q_vel, cn, ct, normal_vectors, tangent_vectors, proj_area)

torque_d = calc_torque_d(com_vectors, force_d)
#    torques_d.append(torque_d)

force_s = calc_force_s(P, proj_area, cos_theta, diffuse, ray_direction, spec_, normal_vectors)
 #   forces_s.append(force_s)

torque_s = calc_torque_s(com_vectors, force_s)
 #   torques_s.append(torque_s)
    
        
F_d = sum(force_d)   
T_d = sum(torque_d)   
F_s = sum(force_s)   
T_s = sum(torque_s)

    # pyvista_ndarray([ 1.30906381e-20,  6.15865957e-20, -8.04264484e-20])
#%%
#real drag force

A = 2*np.pi * (0.1)**2 #radius = 0.1m # area is half surface area of sphere
v = ray_direction*v_mag
F_R = 0.5 * den * v_mag * v * A

print(F_R, F_d)

error = np.linalg.norm(F_d - F_R)/np.linalg.norm(F_R)

