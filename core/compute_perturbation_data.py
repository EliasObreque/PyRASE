# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 18:08:53 2025

@author: mndc5
"""


import pickle
from tqdm import tqdm

import numpy as np
import os

from core.monitor import show_ray_tracing_fast
from core.optimal_ray_tracing import compute_ray_tracing_fast_optimized
from core.drag_models import (compute_fmf_drag_model, compute_analytical_prism_coefficients,
                              compute_coefficients_schaaf, get_atmospheric_condition, get_sphere_drag_coefficient)
from core.srp_models import compute_srp_lambert_model
kB = 1.3806488e-23 # J/K

def compute_ann_data(mesh, sphere_vectors:np.ndarray, sim_data: dict,
                     filename: str, force: bool,
                     res_x:int = 500, res_y:int = 500, save_analytical=False,
                     save_data=True, reference_geometry="Sphere"):
    
    data_mesh = {"r_": [],
                 #"mesh_results": [],
                 "F_drag": [],
                 "T_drag": [],
                 "CA_model": [],
                 "CN_model": [],
                 "CS_model": [],
                 "CA_analytical": [],
                 "CN_analytical": [],
                 "CS_analytical": [],
                 "CA_error": [],
                 "CN_error": [],
                 "CS_error": [],
                 "F_srp": [],
                 "T_srp": [],
                 "Cr_CB": [],
                 "Cr_model": [],
                 "Cd_sphere": [],
                 "Cd_model": [],
                 "Cr_error": []
                 }
    
    if not os.path.exists(filename) or force:
        print("Computing data ...")
        
        # SRP
        spec    = sim_data["spec_srp"]
        diffuse = sim_data["diffuse_srp"]
        # DRAG
        v_inf   = sim_data["v_inf"]
        alt_km  = sim_data["alt_km"]
        time_str= sim_data["time_str"]
        sigma_N = sim_data["sigma_N"]
        sigma_T = sim_data["sigma_T"]
        # T_inf   = sim_data["T_inf"]
        T_wall  = sim_data["T_wall"]
        A_ref   = sim_data["A_ref"]
        
        lx = sim_data.get("lx", 1)
        ly = sim_data.get("ly", 1)
        lz = sim_data.get("lz", 1)
        
        
        
        T_inf, rho, m_particle, r_specific = get_atmospheric_condition(time_str,
                                                                       alt_km)
    
        P_SRP = 4.56e-6
        q_inf = 0.5 * rho * v_inf ** 2 
        
        sim_data["q_inf"] = q_inf
        sim_data["rho"] = rho
        sim_data["T_inf"] = T_inf
        sim_data["m_particle"] = m_particle
        sim_data["P_srp"] = P_SRP
        data_mesh["sim_data"] = sim_data
        
        Cr = 1 + spec + 2*diffuse/3
        vm = np.sqrt(2.0 * kB * T_inf / m_particle)
        N_SAMPLE = len(sphere_vectors)
        
        for r_inout in tqdm(sphere_vectors, "Computation of perturbation", N_SAMPLE):
            res_prop = compute_ray_tracing_fast_optimized(mesh, r_inout, res_x, res_y, verbose=0)

            F_drag_total, T_drag_total, F_srp_total, T_srp_total = compute_ray_perturbation_step(res_prop, sim_data)

            C_A = F_drag_total[0] / (q_inf * A_ref)
            C_N = F_drag_total[2] / (q_inf * A_ref)
            C_S = F_drag_total[1] / (q_inf * A_ref)

            # ERROR
            if save_analytical:
                ca, cs, cn, _ = compute_analytical_prism_coefficients(
                    lx, ly, lz, r_inout * v_inf, sigma_N, sigma_T,
                    T_inf, T_wall, m_particle, A_ref
                )

                error_ca = C_A - ca
                error_cn = C_N - cn
                error_cs = C_S - cs
            else:
                ca = cn = cs = 99
                error_ca = error_cn = error_cs = 99

            A_ref = res_prop.get("A_fem_proj", A_ref)
            Cr_model = np.linalg.norm(F_srp_total) / A_ref / P_SRP
            T_ratio = (T_wall / T_inf)
            s = (v_inf / vm)
            Cd_sphere = get_sphere_drag_coefficient(s, T_ratio, sigma_N, sigma_T)
            Cd_model = np.linalg.norm(F_drag_total) / A_ref / q_inf
            print(f"\n Currents DRAG values. CD sphere: {Cd_sphere} - CD model: {Cd_model}", A_ref)
            print(f"\n Currents SRP values. CR sphere: {Cr} - CR model: {Cr_model}", A_ref)
            if save_analytical:
                error_cr = Cr_model - Cr
                print(f"\nCoefficients errors: Ca = {error_ca}, Cn = {error_cn}, Cs = {error_cs}, Cr = {error_cr}")

                if np.abs(error_ca) > 0.2 or np.abs(error_cn) > 0.2 or np.abs(error_cs) > 0.2:
                    show_ray_tracing_fast(mesh, res_prop, filename="", show_mesh=True, save_3d=False)
            else:
                error_cr = 99
            data_mesh["r_"].append(r_inout)
            # data_mesh["mesh_results"].append(res_prop)
            # drag
            data_mesh["F_drag"].append(F_drag_total)
            data_mesh["T_drag"].append(T_drag_total)
            data_mesh["CA_model"].append(C_A)
            data_mesh["CN_model"].append(C_N)
            data_mesh["CS_model"].append(C_S)
            
            data_mesh["CA_analytical"].append(ca)
            data_mesh["CN_analytical"].append(cs)
            data_mesh["CS_analytical"].append(cn)
            data_mesh["CA_error"].append(error_ca)
            data_mesh["CN_error"].append(error_cn)
            data_mesh["CS_error"].append(error_cs)
            #srp
            data_mesh["F_srp"].append(F_srp_total)
            data_mesh["T_srp"].append(T_srp_total)
            data_mesh["Cr_CB"].append(Cr)
            data_mesh["Cr_model"].append(Cr_model)
            data_mesh["Cd_sphere"].append(Cd_sphere)
            data_mesh["Cd_model"].append(Cd_model)
            data_mesh["Cr_error"].append(error_cr)
            print("Force:", F_srp_total, F_drag_total)
        if save_data:
            with open(filename, "wb") as file_:
                pickle.dump(data_mesh, file_)
    else:
        with open(filename, "rb") as file_:
            data_mesh = pickle.load(file_)
            
    return data_mesh


def compute_ray_perturbation_step(res_prop, sim_data):
    # SRP
    spec = sim_data["spec_srp"]
    diffuse = sim_data["diffuse_srp"]
    # DRAG
    v_inf = sim_data["v_inf"]
    alt_km = sim_data["alt_km"]
    time_str = sim_data["time_str"]
    sigma_N = sim_data["sigma_N"]
    sigma_T = sim_data["sigma_T"]
    # T_inf   = sim_data["T_inf"]
    T_wall = sim_data["T_wall"]

    hits = res_prop['hit_points']  # (N,3), m
    ray_dir = res_prop['r_source']
    cos_th = res_prop['cos_th']  # cos(θ) where θ is angle between ray and normal
    normal_cell = res_prop['cell_normal']  # (N,3)
    # A_fem = res_prop['A_fem']
    Area_r = res_prop['area_proj']

    # '2014-07-22 22:18:45'  # time(UTC)
    T_inf, rho, m_particle, r_specific = get_atmospheric_condition(time_str, alt_km)

    P_SRP = 4.56e-6
    q_inf = 0.5 * rho * v_inf ** 2

    cn, ct = compute_coefficients_schaaf(cos_th, v_inf, sigma_N, sigma_T, T_inf, T_wall, m_particle)

    F_drag_total, T_drag_total, F_d, T_d = compute_fmf_drag_model(q_inf, ray_dir, normal_cell,
                                                                  hits, Area_r, cn, ct)

    F_srp_total, T_srp_total, F_s, T_s = compute_srp_lambert_model(res_prop,
                                                                   ray_dir,
                                                                   np.zeros(3),
                                                                   diffuse,
                                                                   spec,
                                                                   P_SRP)

    return F_drag_total, T_drag_total, F_srp_total, T_srp_total




    