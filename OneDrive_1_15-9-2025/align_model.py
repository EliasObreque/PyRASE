# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 14:27:16 2025

@author: lushi
"""
import pyvista as pv
import numpy as np
import vtk 
import time
from scipy.special import erf

f_x_d = []
f_y_d = []
f_z_d = []

t_x_d = []
t_y_d = []
t_z_d = []

f_x_s =[]
f_y_s = []
f_z_s = []

t_x_s = []
t_y_s = []
t_z_s = []

v_x =[]
v_y =[]
v_z =[]

n_alpha = 5
n_theta = 5

theta = np.linspace(0, 2 * np.pi, n_theta)
alpha = np.linspace(-np.pi/2, np.pi/2, n_alpha)
theta, alpha = np.meshgrid(theta, alpha)

# sphere coords
x = -1 * np.sin(alpha) * np.cos(theta) # inward pointing
y = -1 * np.sin(alpha) * np.sin(theta)
z =  -1 * np.cos(alpha)

sphere_vectors = np.stack((x, y, z), axis=2)
sphere_vectors = sphere_vectors.reshape(-1, 3)

mesh = pv.read("ALIGN-6U.STL") 

mesh = mesh.triangulate().clean()

mesh.rotate_z(45, inplace=True)
mesh.rotate_x(45,inplace=True)

#ta = time.time() #records time from this point

#%%

for v in sphere_vectors:
    ray_direction = np.array([0,0,-1]) #np.array(v)  #rays shine straight down, negative z
    ray_direction = ray_direction / np.linalg.norm(ray_direction) #normalises sun direction vector

    v_x.append(ray_direction[0])
    v_y.append(ray_direction[1])
    v_z.append(ray_direction[2])
    
    res_x = 50
    res_y = 50

    bounds = mesh.bounds #(xmin, xmax, ymin, ymax, zmin, zmax)

    pixel_width = (max(bounds[1]-bounds[0], bounds[3]-bounds[2])) / res_x # in mm
    pixel_height = (max(bounds[1]-bounds[0], bounds[3]-bounds[2]))/ res_y
    z_start = bounds[5]+500
        
    x_range = np.linspace(bounds[2]-200, bounds[3]+200, res_x)
    y_range = np.linspace(bounds[2]-200, bounds[3]+200, res_y)

    hit_points = []
    ray_starts = []
    ray_ends = []
    
    xx, yy = np.meshgrid(x_range + pixel_width/2, y_range + pixel_height/2)
    zz = np.full_like(xx, z_start)

    starts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    directions = np.tile(ray_direction, (starts.shape[0], 1))

    # Perform multi-ray trace
    points, cell_ids, ray_ids = mesh.multi_ray_trace(starts, directions, first_point=True)

    # points: all hit points
    # ind: indices of start rays (corresponds to which start each hit came from)

    # Optional: filter unique hit points
    hit_points = [tuple(pt) for pt in points]

    for y in y_range:
            for x in x_range:
                start = [x+pixel_width/2, y+pixel_height/2, z_start]  # Start above the mesh
                end = [x+pixel_width/2, y+pixel_height/2, z_start-3*z_start]  # End point
                ray_starts.append(start)
                ray_ends.append(end)
                
                # Perform ray trace
                points, _ = mesh.ray_trace(start, end, first_point=True)
                
                if len(points) == 3:
                    points = points.reshape(-1, 3)
                    hit_points.append(points[0])  # store first hit point
                else:
                    hit_points.append(None)  # No intersection
                    
    hit_points = [hit for hit in hit_points if hit is not None] # gets rid of missed rays
  #  hit_points = [tuple(coord) for coord in hit_points]  # keeps only coords

    #now i have coords where the rays hit

    #%%

    #finding illuminated cells 

    vtk_mesh = mesh.GetDataSet() if hasattr(mesh, "GetDataSet") else mesh #convert mesh to vtk

 # Create and build vtkcell locator
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(vtk_mesh)
    locator.BuildLocator()

#    illuminated_cells = set()
#   for pt in hit_points:
#        cid = locator.FindCell(pt)
#        if cid != -1: #not null
#            illuminated_cells.add(cid)
        
    cell_ids_per_ray = []
    hit_points_per_ray = []

    for pt in hit_points:
        cid = locator.FindCell(pt)
        if cid != -1:
            cell_ids_per_ray.append(cid)
            hit_points_per_ray.append(pt)


    #%%

    normal_vectors = mesh.cell_normals[cell_ids_per_ray]  # array of normals, repeated cells included
 #   normal_vectors = mesh.cell_normals[illuminated_cells]

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

    #%%
    #now find area of each projected pixel

    pixel_area = pixel_width*pixel_height *(0.001 * 0.001) #converts from mm to m
    cos_theta = np.dot(normal_vectors, ray_direction)
    proj_area = pixel_area/cos_theta 

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

    for n_ , t_ , c_, area_ in zip(normal_vectors, tangent_vectors, com_vectors, proj_area):
        
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
    f_x_d.append(F_d[0])
    f_y_d.append(F_d[1])
    f_z_d.append(F_d[2])

    T_d = sum(torques_d)
    t_x_d.append(T_d[0])
    t_y_d.append(T_d[1])
    t_z_d.append(T_d[2])
    
    F_s = sum(forces_s)
    f_x_s.append(F_s[0])
    f_y_s.append(F_s[1])
    f_z_s.append(F_s[2])
    
    T_s = sum(torques_s)
    t_x_s.append(T_s[0])
    t_y_s.append(T_s[1])
    t_z_s.append(T_s[2])


#%%
#create dictionary for data and convert to csv file
import pandas as pd
data={'v_x': v_x, 'v_y': v_y, 'v_z':v_z, 'f_x_d': f_x_d, 'f_y_d': f_y_d, 'f_z_d':f_z_d, 't_x_d':t_x_d, 't_y_d':t_y_d, 't_z_d': t_z_d, 'f_x_s':f_x_s, 'f_y_s':f_y_s, 'f_z_s':f_z_s, 't_x_s':t_x_s, 't_y_s':t_y_s, 't_z_s':t_z_s}
df = pd.DataFrame(data)
df.to_csv('data.csv')

#%%

#table of values for drag

import matplotlib.pyplot as plt

data_d = list(zip(v_x, v_y, v_z, f_x_d, f_y_d, f_z_d, t_x_d, t_y_d, t_z_d))

fig, ax = plt.subplots()
ax.axis('off')  # hide axes

table_d = ax.table(cellText=data_d, colLabels=["v_x", "v_y", "v_z", "f_x_d", "f_y_d", "f_z_d", "t_x_d", "t_y_d", "t_z_d"], loc='center', cellLoc='center')

table_d.auto_set_font_size(False)
table_d.set_fontsize(20)
table_d.scale(8, 8)

# Show the plot
plt.show()

#%%

#table of values for srp

data_s = list(zip(v_x, v_y, v_z, f_x_s, f_y_s, f_z_s, t_x_s, t_y_s, t_z_s))

fig, ax = plt.subplots()
ax.axis('off')  # hide axes

table_s = ax.table(cellText=data_s, colLabels=["v_x", "v_y", "v_z", "f_x_s", "f_y_s", "f_z_s", "t_x_s", "t_y_s", "t_z_s"], loc='center', cellLoc='center')

# Optional styling
table_s.auto_set_font_size(False)
table_s.set_fontsize(20)
table_s.scale(8, 8)

# Show the plot
plt.show()

#%%
#PLOT
#now visualise rays from pixel array onto satellite

pixel_grid = pv.StructuredGrid()
pixel_grid.dimensions = (res_x, res_y, 1)  # 2D grid
xx, yy = np.meshgrid(x_range, y_range)
pixel_grid.points = np.column_stack((xx.ravel(), yy.ravel(), np.full(res_x*res_y, z_start)))
plotter = pv.Plotter()

plotter.add_mesh(mesh, show_edges=True) #add mesh
plotter.add_mesh(pixel_grid, style="wireframe", color="green", line_width=2)
plotter.add_mesh(np.array(hit_points), color="red", point_size=5)

#for start, end in zip(ray_starts, ray_ends):   
#    line = pv.Line(start, end)
#    plotter.add_mesh(line, color="blue", line_width=1)
    
for start, end in zip(ray_starts, ray_ends):
    points, _ = mesh.ray_trace(start, end)
    if points.size > 0:
       line = pv.Line(start, points[0])
       plotter.add_mesh(line, color="blue", line_width=1)

#centers = np.array(ray_starts)
#plotter.add_mesh(centers, color="red", point_size=3)

plotter.show()


