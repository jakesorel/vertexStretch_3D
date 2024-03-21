import sys
sys.dont_write_bytecode = True
import os
SCRIPT_DIR = "../../../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

from vertex_stretch_3d_periodic.simulation import Simulation
from vertex_stretch_3d_periodic.mesh import assemble_scalar
import numpy as np
import networkx as nx
from scipy import sparse
import pandas as pd
import pickle
import bz2
import os
from joblib import Parallel, delayed
import sys

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def run_simulation(T_cortical,alpha,A0,p_notch,file_name,index):
    try:
        mesh_options = {"L": 18.4, "A_init": 2., "V_init": 1., "init_noise": 1e-1, "eps": 0.002, "l_mult": 1.05,
                        "seed":index+2024}
        tissue_options = {"kappa_A": (0.02, 0.02),
                          "A0": (A0,A0),
                          "T_lateral": (0., 0. * alpha),
                          "T_cortical": (T_cortical,T_cortical * alpha),
                          "T_basal": (0., 0.),
                          "F_bend": (1., 1.),
                          "T_external": 0.,
                          "kappa_V": (3.0, 3.0),
                          "V0": (1.0, 1.0),
                          "p_notch": p_notch,
                          "mu_L": 3.,
                          "L_min": 2.}
        simulation_options = {"dt": 0.01,
                              "tfin": 50.,
                              "t_skip": 10}
        sim = Simulation(simulation_options, tissue_options, mesh_options)

        sim.simulate()
        export_terminal(sim,index,file_name)
        out_dict = extract_statistics(sim)
        export_terminal(sim,index,file_name)
    except:
        keys = list(["index",'L_equilib', 'total_positive_by_vertex_0', 'total_positive_by_vertex_1', 'total_positive_by_vertex_2', 'total_positive_by_vertex_3', 'A_av', 'A_P_av', 'A_N_av', 'A_std', 'A_P_std', 'A_N_std', 'P_av', 'P_P_av', 'P_N_av', 'P_std', 'P_P_std', 'P_N_std', 'H_av', 'H_P_av', 'H_N_av', 'H_std', 'H_P_std', 'H_N_std', 'ruggedness', 'z_by_positive_av_0', 'z_by_positive_av_1', 'z_by_positive_av_2', 'z_by_positive_av_3', 'z_by_positive_std_0', 'z_by_positive_std_1', 'z_by_positive_std_2', 'z_by_positive_std_3', 'n_notch_neighbours_count_0', 'n_notch_neighbours_count_1', 'n_notch_neighbours_count_2', 'n_notch_neighbours_count_3', 'n_notch_neighbours_count_4', 'n_notch_neighbours_count_5', 'n_notch_neighbours_count_6', 'n_notch_neighbours_count_7', 'n_notch_neighbours_count_8', 'n_notch_neighbours_count_9', 'n_notch_neighbours_count_P_0', 'n_notch_neighbours_count_P_1', 'n_notch_neighbours_count_P_2', 'n_notch_neighbours_count_P_3', 'n_notch_neighbours_count_N_0', 'n_notch_neighbours_count_N_1', 'n_notch_neighbours_count_N_2', 'n_notch_neighbours_count_N_3', 'A_by_notch_neighbours_av_0', 'A_by_notch_neighbours_av_1', 'A_by_notch_neighbours_av_2', 'A_by_notch_neighbours_av_3', 'A_by_notch_neighbours_av_4', 'A_by_notch_neighbours_av_5', 'A_by_notch_neighbours_av_6', 'A_by_notch_neighbours_av_7', 'A_by_notch_neighbours_av_8', 'A_by_notch_neighbours_av_9', 'A_by_notch_neighbours_std_0', 'A_by_notch_neighbours_std_1', 'A_by_notch_neighbours_std_2', 'A_by_notch_neighbours_std_3', 'A_by_notch_neighbours_std_4', 'A_by_notch_neighbours_std_5', 'A_by_notch_neighbours_std_6', 'A_by_notch_neighbours_std_7', 'A_by_notch_neighbours_std_8', 'A_by_notch_neighbours_std_9', 'P_by_notch_neighbours_av_0', 'P_by_notch_neighbours_av_1', 'P_by_notch_neighbours_av_2', 'P_by_notch_neighbours_av_3', 'P_by_notch_neighbours_av_4', 'P_by_notch_neighbours_av_5', 'P_by_notch_neighbours_av_6', 'P_by_notch_neighbours_av_7', 'P_by_notch_neighbours_av_8', 'P_by_notch_neighbours_av_9', 'P_by_notch_neighbours_std_0', 'P_by_notch_neighbours_std_1', 'P_by_notch_neighbours_std_2', 'P_by_notch_neighbours_std_3', 'P_by_notch_neighbours_std_4', 'P_by_notch_neighbours_std_5', 'P_by_notch_neighbours_std_6', 'P_by_notch_neighbours_std_7', 'P_by_notch_neighbours_std_8', 'P_by_notch_neighbours_std_9', 'H_by_notch_neighbours_av_0', 'H_by_notch_neighbours_av_1', 'H_by_notch_neighbours_av_2', 'H_by_notch_neighbours_av_3', 'H_by_notch_neighbours_av_4', 'H_by_notch_neighbours_av_5', 'H_by_notch_neighbours_av_6', 'H_by_notch_neighbours_av_7', 'H_by_notch_neighbours_av_8', 'H_by_notch_neighbours_av_9', 'H_by_notch_neighbours_std_0', 'H_by_notch_neighbours_std_1', 'H_by_notch_neighbours_std_2', 'H_by_notch_neighbours_std_3', 'H_by_notch_neighbours_std_4', 'H_by_notch_neighbours_std_5', 'H_by_notch_neighbours_std_6', 'H_by_notch_neighbours_std_7', 'H_by_notch_neighbours_std_8', 'H_by_notch_neighbours_std_9', 'A_by_notch_neighbours_P_av_0', 'A_by_notch_neighbours_P_av_1', 'A_by_notch_neighbours_P_av_2', 'A_by_notch_neighbours_P_av_3', 'A_by_notch_neighbours_P_av_4', 'A_by_notch_neighbours_P_av_5', 'A_by_notch_neighbours_P_av_6', 'A_by_notch_neighbours_P_av_7', 'A_by_notch_neighbours_P_av_8', 'A_by_notch_neighbours_P_av_9', 'A_by_notch_neighbours_P_std_0', 'A_by_notch_neighbours_P_std_1', 'A_by_notch_neighbours_P_std_2', 'A_by_notch_neighbours_P_std_3', 'A_by_notch_neighbours_P_std_4', 'A_by_notch_neighbours_P_std_5', 'A_by_notch_neighbours_P_std_6', 'A_by_notch_neighbours_P_std_7', 'A_by_notch_neighbours_P_std_8', 'A_by_notch_neighbours_P_std_9', 'P_by_notch_neighbours_P_av_0', 'P_by_notch_neighbours_P_av_1', 'P_by_notch_neighbours_P_av_2', 'P_by_notch_neighbours_P_av_3', 'P_by_notch_neighbours_P_av_4', 'P_by_notch_neighbours_P_av_5', 'P_by_notch_neighbours_P_av_6', 'P_by_notch_neighbours_P_av_7', 'P_by_notch_neighbours_P_av_8', 'P_by_notch_neighbours_P_av_9', 'P_by_notch_neighbours_P_std_0', 'P_by_notch_neighbours_P_std_1', 'P_by_notch_neighbours_P_std_2', 'P_by_notch_neighbours_P_std_3', 'P_by_notch_neighbours_P_std_4', 'P_by_notch_neighbours_P_std_5', 'P_by_notch_neighbours_P_std_6', 'P_by_notch_neighbours_P_std_7', 'P_by_notch_neighbours_P_std_8', 'P_by_notch_neighbours_P_std_9', 'H_by_notch_neighbours_P_av_0', 'H_by_notch_neighbours_P_av_1', 'H_by_notch_neighbours_P_av_2', 'H_by_notch_neighbours_P_av_3', 'H_by_notch_neighbours_P_av_4', 'H_by_notch_neighbours_P_av_5', 'H_by_notch_neighbours_P_av_6', 'H_by_notch_neighbours_P_av_7', 'H_by_notch_neighbours_P_av_8', 'H_by_notch_neighbours_P_av_9', 'H_by_notch_neighbours_P_std_0', 'H_by_notch_neighbours_P_std_1', 'H_by_notch_neighbours_P_std_2', 'H_by_notch_neighbours_P_std_3', 'H_by_notch_neighbours_P_std_4', 'H_by_notch_neighbours_P_std_5', 'H_by_notch_neighbours_P_std_6', 'H_by_notch_neighbours_P_std_7', 'H_by_notch_neighbours_P_std_8', 'H_by_notch_neighbours_P_std_9', 'A_by_notch_neighbours_N_av_0', 'A_by_notch_neighbours_N_av_1', 'A_by_notch_neighbours_N_av_2', 'A_by_notch_neighbours_N_av_3', 'A_by_notch_neighbours_N_av_4', 'A_by_notch_neighbours_N_av_5', 'A_by_notch_neighbours_N_av_6', 'A_by_notch_neighbours_N_av_7', 'A_by_notch_neighbours_N_av_8', 'A_by_notch_neighbours_N_av_9', 'A_by_notch_neighbours_N_std_0', 'A_by_notch_neighbours_N_std_1', 'A_by_notch_neighbours_N_std_2', 'A_by_notch_neighbours_N_std_3', 'A_by_notch_neighbours_N_std_4', 'A_by_notch_neighbours_N_std_5', 'A_by_notch_neighbours_N_std_6', 'A_by_notch_neighbours_N_std_7', 'A_by_notch_neighbours_N_std_8', 'A_by_notch_neighbours_N_std_9', 'P_by_notch_neighbours_N_av_0', 'P_by_notch_neighbours_N_av_1', 'P_by_notch_neighbours_N_av_2', 'P_by_notch_neighbours_N_av_3', 'P_by_notch_neighbours_N_av_4', 'P_by_notch_neighbours_N_av_5', 'P_by_notch_neighbours_N_av_6', 'P_by_notch_neighbours_N_av_7', 'P_by_notch_neighbours_N_av_8', 'P_by_notch_neighbours_N_av_9', 'P_by_notch_neighbours_N_std_0', 'P_by_notch_neighbours_N_std_1', 'P_by_notch_neighbours_N_std_2', 'P_by_notch_neighbours_N_std_3', 'P_by_notch_neighbours_N_std_4', 'P_by_notch_neighbours_N_std_5', 'P_by_notch_neighbours_N_std_6', 'P_by_notch_neighbours_N_std_7', 'P_by_notch_neighbours_N_std_8', 'P_by_notch_neighbours_N_std_9', 'H_by_notch_neighbours_N_av_0', 'H_by_notch_neighbours_N_av_1', 'H_by_notch_neighbours_N_av_2', 'H_by_notch_neighbours_N_av_3', 'H_by_notch_neighbours_N_av_4', 'H_by_notch_neighbours_N_av_5', 'H_by_notch_neighbours_N_av_6', 'H_by_notch_neighbours_N_av_7', 'H_by_notch_neighbours_N_av_8', 'H_by_notch_neighbours_N_av_9', 'H_by_notch_neighbours_N_std_0', 'H_by_notch_neighbours_N_std_1', 'H_by_notch_neighbours_N_std_2', 'H_by_notch_neighbours_N_std_3', 'H_by_notch_neighbours_N_std_4', 'H_by_notch_neighbours_N_std_5', 'H_by_notch_neighbours_N_std_6', 'H_by_notch_neighbours_N_std_7', 'H_by_notch_neighbours_N_std_8', 'H_by_notch_neighbours_N_std_9', 'mean_shortest_path', 'geom_mean_shortest_path', 'median_shortest_path', 'mean_shortest_path_P', 'geom_mean_shortest_path_P', 'median_shortest_path_P', 'mean_shortest_path_N', 'geom_mean_shortest_path_N', 'median_shortest_path_N', 'mean_cluster_size', 'geom_mean_cluster_size', 'median_cluster_size', 'mean_cluster_size_P', 'geom_mean_cluster_size_P', 'median_cluster_size_P', 'mean_cluster_size_N', 'geom_mean_cluster_size_N', 'median_cluster_size_N', 'A_by_shortest_path_av_1', 'A_by_shortest_path_av_2', 'A_by_shortest_path_av_3', 'A_by_shortest_path_av_4', 'A_by_shortest_path_av_5', 'A_by_shortest_path_av_6', 'A_by_shortest_path_std_1', 'A_by_shortest_path_std_2', 'A_by_shortest_path_std_3', 'A_by_shortest_path_std_4', 'A_by_shortest_path_std_5', 'A_by_shortest_path_std_6', 'P_by_shortest_path_av_1', 'P_by_shortest_path_av_2', 'P_by_shortest_path_av_3', 'P_by_shortest_path_av_4', 'P_by_shortest_path_av_5', 'P_by_shortest_path_av_6', 'P_by_shortest_path_std_1', 'P_by_shortest_path_std_2', 'P_by_shortest_path_std_3', 'P_by_shortest_path_std_4', 'P_by_shortest_path_std_5', 'P_by_shortest_path_std_6', 'H_by_shortest_path_av_1', 'H_by_shortest_path_av_2', 'H_by_shortest_path_av_3', 'H_by_shortest_path_av_4', 'H_by_shortest_path_av_5', 'H_by_shortest_path_av_6', 'H_by_shortest_path_std_1', 'H_by_shortest_path_std_2', 'H_by_shortest_path_std_3', 'H_by_shortest_path_std_4', 'H_by_shortest_path_std_5', 'H_by_shortest_path_std_6', 'A_by_shortest_path_P_av_1', 'A_by_shortest_path_P_av_2', 'A_by_shortest_path_P_av_3', 'A_by_shortest_path_P_av_4', 'A_by_shortest_path_P_av_5', 'A_by_shortest_path_P_av_6', 'A_by_shortest_path_P_std_1', 'A_by_shortest_path_P_std_2', 'A_by_shortest_path_P_std_3', 'A_by_shortest_path_P_std_4', 'A_by_shortest_path_P_std_5', 'A_by_shortest_path_P_std_6', 'P_by_shortest_path_P_av_1', 'P_by_shortest_path_P_av_2', 'P_by_shortest_path_P_av_3', 'P_by_shortest_path_P_av_4', 'P_by_shortest_path_P_av_5', 'P_by_shortest_path_P_av_6', 'P_by_shortest_path_P_std_1', 'P_by_shortest_path_P_std_2', 'P_by_shortest_path_P_std_3', 'P_by_shortest_path_P_std_4', 'P_by_shortest_path_P_std_5', 'P_by_shortest_path_P_std_6', 'H_by_shortest_path_P_av_1', 'H_by_shortest_path_P_av_2', 'H_by_shortest_path_P_av_3', 'H_by_shortest_path_P_av_4', 'H_by_shortest_path_P_av_5', 'H_by_shortest_path_P_av_6', 'H_by_shortest_path_P_std_1', 'H_by_shortest_path_P_std_2', 'H_by_shortest_path_P_std_3', 'H_by_shortest_path_P_std_4', 'H_by_shortest_path_P_std_5', 'H_by_shortest_path_P_std_6', 'A_by_shortest_path_N_av_1', 'A_by_shortest_path_N_av_2', 'A_by_shortest_path_N_av_3', 'A_by_shortest_path_N_av_4', 'A_by_shortest_path_N_av_5', 'A_by_shortest_path_N_av_6', 'A_by_shortest_path_N_std_1', 'A_by_shortest_path_N_std_2', 'A_by_shortest_path_N_std_3', 'A_by_shortest_path_N_std_4', 'A_by_shortest_path_N_std_5', 'A_by_shortest_path_N_std_6', 'P_by_shortest_path_N_av_1', 'P_by_shortest_path_N_av_2', 'P_by_shortest_path_N_av_3', 'P_by_shortest_path_N_av_4', 'P_by_shortest_path_N_av_5', 'P_by_shortest_path_N_av_6', 'P_by_shortest_path_N_std_1', 'P_by_shortest_path_N_std_2', 'P_by_shortest_path_N_std_3', 'P_by_shortest_path_N_std_4', 'P_by_shortest_path_N_std_5', 'P_by_shortest_path_N_std_6', 'H_by_shortest_path_N_av_1', 'H_by_shortest_path_N_av_2', 'H_by_shortest_path_N_av_3', 'H_by_shortest_path_N_av_4', 'H_by_shortest_path_N_av_5', 'H_by_shortest_path_N_av_6', 'H_by_shortest_path_N_std_1', 'H_by_shortest_path_N_std_2', 'H_by_shortest_path_N_std_3', 'H_by_shortest_path_N_std_4', 'H_by_shortest_path_N_std_5', 'H_by_shortest_path_N_std_6', 'A_by_n_cc_av_1', 'A_by_n_cc_av_2', 'A_by_n_cc_av_3', 'A_by_n_cc_av_4', 'A_by_n_cc_av_5', 'A_by_n_cc_av_6', 'A_by_n_cc_av_7', 'A_by_n_cc_av_8', 'A_by_n_cc_std_1', 'A_by_n_cc_std_2', 'A_by_n_cc_std_3', 'A_by_n_cc_std_4', 'A_by_n_cc_std_5', 'A_by_n_cc_std_6', 'A_by_n_cc_std_7', 'A_by_n_cc_std_8', 'P_by_n_cc_av_1', 'P_by_n_cc_av_2', 'P_by_n_cc_av_3', 'P_by_n_cc_av_4', 'P_by_n_cc_av_5', 'P_by_n_cc_av_6', 'P_by_n_cc_av_7', 'P_by_n_cc_av_8', 'P_by_n_cc_std_1', 'P_by_n_cc_std_2', 'P_by_n_cc_std_3', 'P_by_n_cc_std_4', 'P_by_n_cc_std_5', 'P_by_n_cc_std_6', 'P_by_n_cc_std_7', 'P_by_n_cc_std_8', 'H_by_n_cc_av_1', 'H_by_n_cc_av_2', 'H_by_n_cc_av_3', 'H_by_n_cc_av_4', 'H_by_n_cc_av_5', 'H_by_n_cc_av_6', 'H_by_n_cc_av_7', 'H_by_n_cc_av_8', 'H_by_n_cc_std_1', 'H_by_n_cc_std_2', 'H_by_n_cc_std_3', 'H_by_n_cc_std_4', 'H_by_n_cc_std_5', 'H_by_n_cc_std_6', 'H_by_n_cc_std_7', 'H_by_n_cc_std_8', 'A_by_n_cc_P_av_1', 'A_by_n_cc_P_av_2', 'A_by_n_cc_P_av_3', 'A_by_n_cc_P_av_4', 'A_by_n_cc_P_av_5', 'A_by_n_cc_P_av_6', 'A_by_n_cc_P_av_7', 'A_by_n_cc_P_av_8', 'A_by_n_cc_P_std_1', 'A_by_n_cc_P_std_2', 'A_by_n_cc_P_std_3', 'A_by_n_cc_P_std_4', 'A_by_n_cc_P_std_5', 'A_by_n_cc_P_std_6', 'A_by_n_cc_P_std_7', 'A_by_n_cc_P_std_8', 'P_by_n_cc_P_av_1', 'P_by_n_cc_P_av_2', 'P_by_n_cc_P_av_3', 'P_by_n_cc_P_av_4', 'P_by_n_cc_P_av_5', 'P_by_n_cc_P_av_6', 'P_by_n_cc_P_av_7', 'P_by_n_cc_P_av_8', 'P_by_n_cc_P_std_1', 'P_by_n_cc_P_std_2', 'P_by_n_cc_P_std_3', 'P_by_n_cc_P_std_4', 'P_by_n_cc_P_std_5', 'P_by_n_cc_P_std_6', 'P_by_n_cc_P_std_7', 'P_by_n_cc_P_std_8', 'H_by_n_cc_P_av_1', 'H_by_n_cc_P_av_2', 'H_by_n_cc_P_av_3', 'H_by_n_cc_P_av_4', 'H_by_n_cc_P_av_5', 'H_by_n_cc_P_av_6', 'H_by_n_cc_P_av_7', 'H_by_n_cc_P_av_8', 'H_by_n_cc_P_std_1', 'H_by_n_cc_P_std_2', 'H_by_n_cc_P_std_3', 'H_by_n_cc_P_std_4', 'H_by_n_cc_P_std_5', 'H_by_n_cc_P_std_6', 'H_by_n_cc_P_std_7', 'H_by_n_cc_P_std_8', 'A_by_n_cc_N_av_1', 'A_by_n_cc_N_av_2', 'A_by_n_cc_N_av_3', 'A_by_n_cc_N_av_4', 'A_by_n_cc_N_av_5', 'A_by_n_cc_N_av_6', 'A_by_n_cc_N_av_7', 'A_by_n_cc_N_av_8', 'A_by_n_cc_N_std_1', 'A_by_n_cc_N_std_2', 'A_by_n_cc_N_std_3', 'A_by_n_cc_N_std_4', 'A_by_n_cc_N_std_5', 'A_by_n_cc_N_std_6', 'A_by_n_cc_N_std_7', 'A_by_n_cc_N_std_8', 'P_by_n_cc_N_av_1', 'P_by_n_cc_N_av_2', 'P_by_n_cc_N_av_3', 'P_by_n_cc_N_av_4', 'P_by_n_cc_N_av_5', 'P_by_n_cc_N_av_6', 'P_by_n_cc_N_av_7', 'P_by_n_cc_N_av_8', 'P_by_n_cc_N_std_1', 'P_by_n_cc_N_std_2', 'P_by_n_cc_N_std_3', 'P_by_n_cc_N_std_4', 'P_by_n_cc_N_std_5', 'P_by_n_cc_N_std_6', 'P_by_n_cc_N_std_7', 'P_by_n_cc_N_std_8', 'H_by_n_cc_N_av_1', 'H_by_n_cc_N_av_2', 'H_by_n_cc_N_av_3', 'H_by_n_cc_N_av_4', 'H_by_n_cc_N_av_5', 'H_by_n_cc_N_av_6', 'H_by_n_cc_N_av_7', 'H_by_n_cc_N_av_8', 'H_by_n_cc_N_std_1', 'H_by_n_cc_N_std_2', 'H_by_n_cc_N_std_3', 'H_by_n_cc_N_std_4', 'H_by_n_cc_N_std_5', 'H_by_n_cc_N_std_6', 'H_by_n_cc_N_std_7', 'H_by_n_cc_N_std_8'])

        out_dict = dict(zip(keys,np.ones(len(keys))*np.nan))
        out_dict["index"] = index
    return out_dict


def export_terminal(sim,index,file_name):
    tot_props = {"tissue_params":sim.t.tissue_params,
                 "mesh_props":sim.t.mesh.mesh_props}
    with bz2.BZ2File('%s/%d.pbz2'%(file_name,index), 'wb') as f:
        pickle.dump(tot_props, f)


def shortest_path_to_label(graph, start_node, label):
    shortest_path = None
    shortest_distance = float('inf')

    # Find the shortest path to any node with the given label
    for node in graph.nodes():
        if graph.nodes[node]['label'] == label:
            path = nx.shortest_path(graph, start_node, node)
            distance = len(path) - 1  # Length of the path is one less than the number of nodes
            if distance < shortest_distance:
                shortest_path = path
                shortest_distance = distance

    return shortest_path

def get_graph_props(sim):
    self = sim.t.mesh
    is_notch = sim.t.tissue_params["is_notch"]
    edges = np.row_stack(
        [np.column_stack([self.mesh_props["tri"][:, i], self.mesh_props["tri"][:, (i + 1) % 3]]) for i in
         range(3)])

    ##Calculate the shortest path from notch negative to notch positive and vice versa
    graph = nx.Graph()
    graph.add_edges_from(edges)
    nx.set_node_attributes(graph,
                           dict(zip(np.arange(np.max(edges) + 1), is_notch.astype(int))),
                           'label')

    shortest_paths = []
    for i in np.arange(np.max(edges) + 1):
        shortest_paths.append(len(shortest_path_to_label(graph, i, 1 - is_notch.astype(int)[i]))-1)
    shortest_paths = np.array(shortest_paths)

    ##Calculate the adjacency matrix
    adj = sparse.coo_matrix((np.ones(len(edges), dtype=int), (edges[:, 0], edges[:, 1])),
                                               shape=(int(self.mesh_props["n_c"]), int(self.mesh_props["n_c"])))

    is_notch_int = sim.t.tissue_params["is_notch"].astype(int)
    is_notch_edges = is_notch_int[edges]

    adj_notch = sparse.coo_matrix((np.ones((is_notch_edges[:,0] == is_notch_edges[:,1]).sum(),dtype=int), (edges[(is_notch_edges[:,0] == is_notch_edges[:,1]), 0], edges[(is_notch_edges[:,0] == is_notch_edges[:,1]), 1])),
                                               shape=(int(mesh_props["n_c"]), int(mesh_props["n_c"])))

    n_cc, cc_labs = sparse.csgraph.connected_components(adj_notch)


    cluster_size_negative = np.bincount(cc_labs[~is_notch],minlength=int(mesh_props["n_c"]/2))
    cluster_size_positive = np.bincount(cc_labs[is_notch],minlength=int(mesh_props["n_c"]/2))
    return shortest_paths,cc_labs,n_cc,cluster_size_negative,cluster_size_positive


def extract_statistics(sim,index):
    ##L_equilib
    L_equilib = sim.sim_out["L_save"][-1]

    ##Notch_mask
    is_notch = np.array(sim.t.tissue_params["is_notch"])
    t_is_notch = is_notch[sim.t.mesh.mesh_props["tri"]]
    n_positive_by_vertex = np.array(t_is_notch.sum(axis=1))
    total_positive_by_vertex = np.bincount(n_positive_by_vertex,minlength=4)

    ##Areas
    A = np.array(sim.t.mesh.mesh_props["A_apical"])
    A_av = A.mean()
    A_P_av = A[is_notch].mean()
    A_N_av = A[~is_notch].mean()
    A_std = A.std()
    A_P_std = A[is_notch].std()
    A_N_std = A[~is_notch].std()


    ##Perimeters
    P = np.array(sim.t.mesh.mesh_props["P_apical"])
    P_av = P.mean()
    P_P_av = P[is_notch].mean()
    P_N_av = P[~is_notch].mean()
    P_std = P.std()
    P_P_std = P[is_notch].std()
    P_N_std = P[~is_notch].std()



    ##Cell Heights
    H = np.array(sim.t.mesh.mesh_props["R"][:,2])
    H_av = H.mean()
    H_P_av = H[is_notch].mean()
    H_N_av = H[~is_notch].mean()
    H_std = H.std()
    H_P_std = H[is_notch].std()
    H_N_std = H[~is_notch].std()

    ##Ruggedness
    ruggedness = np.log(sim.t.mesh.mesh_props["A_basal"].sum() / sim.t.mesh.mesh_props["A_apical"].sum())


    cell_properties = {"index":index,"L_equilib": L_equilib,
                         "total_positive_by_vertex_0": total_positive_by_vertex[0],
                       "total_positive_by_vertex_1": total_positive_by_vertex[1],
                       "total_positive_by_vertex_2": total_positive_by_vertex[2],
                       "total_positive_by_vertex_3": total_positive_by_vertex[3],
                       "A_av": A_av,
                         "A_P_av": A_P_av,
                         "A_N_av":A_N_av,
                         "A_std":A_std,
                         "A_P_std":A_P_std,
                         "A_N_std":A_N_std,
                         "P_av": P_av,
                         "P_P_av": P_P_av,
                         "P_N_av": P_N_av,
                         "P_std": P_std,
                         "P_P_std": P_P_std,
                         "P_N_std": P_N_std,
                         "H_av": H_av,
                         "H_P_av": H_P_av,
                         "H_N_av": H_N_av,
                         "H_std": H_std,
                         "H_P_std": H_P_std,
                         "H_N_std": H_N_std,
                       "ruggedness":ruggedness
                         }

    ##Vertex Heights
    z = np.array(sim.t.mesh.mesh_props["X"][:,2])
    z_by_positive_av = np.array([z[n_positive_by_vertex==i].mean() for i in range(4)])
    z_by_positive_std = np.array([z[n_positive_by_vertex==i].std() for i in range(4)])

    z_by_positive_dict = dict(zip(["z_by_positive_av_%d"%i for i in range(4)],z_by_positive_av))
    z_by_positive_dict.update(dict(zip(["z_by_positive_std_%d"%i for i in range(4)],z_by_positive_av)))



    ##By number of notch+ neighbours
    tri = sim.t.mesh.mesh_props["tri"]
    n_v = assemble_scalar(np.ones_like(sim.t.mesh.mesh_props["tri"]), tri, int(np.max(tri) + 1))
    n_notch_tri = t_is_notch.astype(int).sum(axis=1)
    n_notch_neighbours = assemble_scalar(np.ones_like(tri) * np.expand_dims(n_notch_tri, 1), tri, int(np.max(tri) + 1))
    n_notch_neighbours = ((n_notch_neighbours - is_notch * n_v) / 2).astype(int)

    n_notch_neighbours_count = np.bincount(n_notch_neighbours,minlength=10)
    n_notch_neighbours_count_P = np.bincount(n_notch_neighbours[is_notch],minlength=10)
    n_notch_neighbours_count_N = np.bincount(n_notch_neighbours[~is_notch],minlength=10)

    n_notch_neighbours_count_dict = dict(zip(["n_notch_neighbours_count_%d"%i for i in range(10)],n_notch_neighbours_count))
    n_notch_neighbours_count_dict.update(dict(zip(["n_notch_neighbours_count_P_%d"%i for i in range(4)],n_notch_neighbours_count_P)))
    n_notch_neighbours_count_dict.update(dict(zip(["n_notch_neighbours_count_N_%d"%i for i in range(4)],n_notch_neighbours_count_N)))



    A_by_notch_neighbours_av = np.array([A[n_notch_neighbours==i].mean() for i in range(10)])
    A_by_notch_neighbours_std = np.array([A[n_notch_neighbours==i].std() for i in range(10)])
    P_by_notch_neighbours_av = np.array([P[n_notch_neighbours==i].mean() for i in range(10)])
    P_by_notch_neighbours_std = np.array([P[n_notch_neighbours==i].std() for i in range(10)])
    H_by_notch_neighbours_av = np.array([H[n_notch_neighbours==i].mean() for i in range(10)])
    H_by_notch_neighbours_std = np.array([H[n_notch_neighbours==i].std() for i in range(10)])


    A_by_notch_neighbours_P_av = np.array([A[(n_notch_neighbours==i)*is_notch].mean() for i in range(10)])
    A_by_notch_neighbours_P_std = np.array([A[(n_notch_neighbours==i)*is_notch].std() for i in range(10)])
    P_by_notch_neighbours_P_av = np.array([P[(n_notch_neighbours==i)*is_notch].mean() for i in range(10)])
    P_by_notch_neighbours_P_std = np.array([P[(n_notch_neighbours==i)*is_notch].std() for i in range(10)])
    H_by_notch_neighbours_P_av = np.array([H[(n_notch_neighbours==i)*is_notch].mean() for i in range(10)])
    H_by_notch_neighbours_P_std = np.array([H[(n_notch_neighbours==i)*is_notch].std() for i in range(10)])

    A_by_notch_neighbours_N_av = np.array([A[(n_notch_neighbours==i)*~is_notch].mean() for i in range(10)])
    A_by_notch_neighbours_N_std = np.array([A[(n_notch_neighbours==i)*~is_notch].std() for i in range(10)])
    P_by_notch_neighbours_N_av = np.array([P[(n_notch_neighbours==i)*~is_notch].mean() for i in range(10)])
    P_by_notch_neighbours_N_std = np.array([P[(n_notch_neighbours==i)*~is_notch].std() for i in range(10)])
    H_by_notch_neighbours_N_av = np.array([H[(n_notch_neighbours==i)*~is_notch].mean() for i in range(10)])
    H_by_notch_neighbours_N_std = np.array([H[(n_notch_neighbours==i)*~is_notch].std() for i in range(10)])

    cell_props_by_n_notch_neighbours_dict = dict(
        zip(["A_by_notch_neighbours_av_%d" % i for i in range(10)], A_by_notch_neighbours_av))
    cell_props_by_n_notch_neighbours_dict.update(
        dict(zip(["A_by_notch_neighbours_std_%d" % i for i in range(10)], A_by_notch_neighbours_std)))
    cell_props_by_n_notch_neighbours_dict.update(
        dict(zip(["P_by_notch_neighbours_av_%d" % i for i in range(10)], P_by_notch_neighbours_av)))
    cell_props_by_n_notch_neighbours_dict.update(
        dict(zip(["P_by_notch_neighbours_std_%d" % i for i in range(10)], P_by_notch_neighbours_std)))
    cell_props_by_n_notch_neighbours_dict.update(
        dict(zip(["H_by_notch_neighbours_av_%d" % i for i in range(10)], H_by_notch_neighbours_av)))
    cell_props_by_n_notch_neighbours_dict.update(dict(zip(["H_by_notch_neighbours_std_%d" % i for i in range(10)],
                 H_by_notch_neighbours_std)))

    cell_props_by_n_notch_neighbours_dict.update(dict(zip(["A_by_notch_neighbours_P_av_%d" % i for i in range(10)],
                 A_by_notch_neighbours_P_av)))
    cell_props_by_n_notch_neighbours_dict.update(dict(zip(["A_by_notch_neighbours_P_std_%d" % i for i in range(10)],
                 A_by_notch_neighbours_P_std)))
    cell_props_by_n_notch_neighbours_dict.update(dict(zip(["P_by_notch_neighbours_P_av_%d" % i for i in range(10)],
                 P_by_notch_neighbours_P_av)))
    cell_props_by_n_notch_neighbours_dict.update(dict(zip(["P_by_notch_neighbours_P_std_%d" % i for i in range(10)],
                 P_by_notch_neighbours_P_std)))
    cell_props_by_n_notch_neighbours_dict.update(dict(zip(["H_by_notch_neighbours_P_av_%d" % i for i in range(10)],
                 H_by_notch_neighbours_P_av)))
    cell_props_by_n_notch_neighbours_dict.update(dict(zip(["H_by_notch_neighbours_P_std_%d" % i for i in range(10)],
                 H_by_notch_neighbours_P_std)))
    cell_props_by_n_notch_neighbours_dict.update(dict(zip(["A_by_notch_neighbours_N_av_%d" % i for i in range(10)],
                 A_by_notch_neighbours_N_av)))
    cell_props_by_n_notch_neighbours_dict.update(dict(zip(["A_by_notch_neighbours_N_std_%d" % i for i in range(10)],
                 A_by_notch_neighbours_N_std)))
    cell_props_by_n_notch_neighbours_dict.update(dict(zip(["P_by_notch_neighbours_N_av_%d" % i for i in range(10)],
                 P_by_notch_neighbours_N_av)))
    cell_props_by_n_notch_neighbours_dict.update(dict(zip(["P_by_notch_neighbours_N_std_%d" % i for i in range(10)],
                 P_by_notch_neighbours_N_std)))
    cell_props_by_n_notch_neighbours_dict.update(dict(zip(["H_by_notch_neighbours_N_av_%d" % i for i in range(10)],
                 H_by_notch_neighbours_N_av)))
    cell_props_by_n_notch_neighbours_dict.update(dict(zip(["H_by_notch_neighbours_N_std_%d" % i for i in range(10)],
                 H_by_notch_neighbours_N_std)))

    ##Graph props
    shortest_paths, cc_labs, n_cc, cluster_size_negative, cluster_size_positive = get_graph_props(sim)
    n_cc_by_cell = np.bincount(cc_labs)[cc_labs]

    ##Truncate the categories for conv.
    n_cc_by_cell_clipped = n_cc_by_cell
    n_cc_by_cell_clipped[n_cc_by_cell_clipped>8] = 8
    shortest_paths_clipped = shortest_paths.copy()
    shortest_paths_clipped[shortest_paths_clipped>6] = 6


    #shortest path props
    mean_shortest_path = np.mean(shortest_paths)
    geom_mean_shortest_path = np.exp(np.mean(np.log(shortest_paths)))
    median_shortest_path = np.median(shortest_paths)

    mean_shortest_path_P = np.mean(shortest_paths[is_notch])
    geom_mean_shortest_path_P = np.exp(np.mean(np.log(shortest_paths[is_notch])))
    median_shortest_path_P = np.median(shortest_paths[is_notch])

    mean_shortest_path_N = np.mean(shortest_paths[~is_notch])
    geom_mean_shortest_path_N = np.exp(np.mean(np.log(shortest_paths[~is_notch])))
    median_shortest_path_N = np.median(shortest_paths[~is_notch])



    #cluster size props
    mean_cluster_size = np.mean(n_cc_by_cell)
    geom_mean_cluster_size = np.exp(np.mean(np.log(n_cc_by_cell)))
    median_cluster_size = np.median(n_cc_by_cell)

    mean_cluster_size_P = np.mean(n_cc_by_cell[is_notch])
    geom_mean_cluster_size_P = np.exp(np.mean(np.log(n_cc_by_cell[is_notch])))
    median_cluster_size_P = np.median(n_cc_by_cell[is_notch])

    mean_cluster_size_N = np.mean(n_cc_by_cell[~is_notch])
    geom_mean_cluster_size_N = np.exp(np.mean(np.log(n_cc_by_cell[~is_notch])))
    median_cluster_size_N = np.median(n_cc_by_cell[~is_notch])

    graph_props_dict = {"mean_shortest_path":mean_shortest_path,
                        "geom_mean_shortest_path":geom_mean_shortest_path,
                        "median_shortest_path":median_shortest_path,
                        "mean_shortest_path_P":mean_shortest_path_P,
                        "geom_mean_shortest_path_P":geom_mean_shortest_path_P,
                        "median_shortest_path_P":median_shortest_path_P,
                        "mean_shortest_path_N":mean_shortest_path_N,
                        "geom_mean_shortest_path_N":geom_mean_shortest_path_N,
                        "median_shortest_path_N":median_shortest_path_N,
                        "mean_cluster_size":mean_cluster_size,
                        "geom_mean_cluster_size":geom_mean_cluster_size,
                        "median_cluster_size":median_cluster_size,
                        "mean_cluster_size_P":mean_cluster_size_P,
                        "geom_mean_cluster_size_P":geom_mean_cluster_size_P,
                        "median_cluster_size_P":median_cluster_size_P,
                        "mean_cluster_size_N":mean_cluster_size_N,
                        "geom_mean_cluster_size_N":geom_mean_cluster_size_N,
                        "median_cluster_size_N":median_cluster_size_N}


    ##Cell properties by graph_props


    A_by_shortest_path_av = np.array([A[shortest_paths_clipped==i].mean() for i in range(1,7)])
    A_by_shortest_path_std = np.array([A[shortest_paths_clipped==i].std() for i in range(1,7)])
    P_by_shortest_path_av = np.array([P[shortest_paths_clipped==i].mean() for i in range(1,7)])
    P_by_shortest_path_std = np.array([P[shortest_paths_clipped==i].std() for i in range(1,7)])
    H_by_shortest_path_av = np.array([H[shortest_paths_clipped==i].mean() for i in range(1,7)])
    H_by_shortest_path_std = np.array([H[shortest_paths_clipped==i].std() for i in range(1,7)])




    A_by_shortest_path_P_av = np.array([A[(shortest_paths_clipped==i)*is_notch].mean() for i in range(1,7)])
    A_by_shortest_path_P_std = np.array([A[(shortest_paths_clipped==i)*is_notch].std() for i in range(1,7)])
    P_by_shortest_path_P_av = np.array([P[(shortest_paths_clipped==i)*is_notch].mean() for i in range(1,7)])
    P_by_shortest_path_P_std = np.array([P[(shortest_paths_clipped==i)*is_notch].std() for i in range(1,7)])
    H_by_shortest_path_P_av = np.array([H[(shortest_paths_clipped==i)*is_notch].mean() for i in range(1,7)])
    H_by_shortest_path_P_std = np.array([H[(shortest_paths_clipped==i)*is_notch].std() for i in range(1,7)])

    A_by_shortest_path_N_av = np.array([A[(shortest_paths_clipped==i)*~is_notch].mean() for i in range(1,7)])
    A_by_shortest_path_N_std = np.array([A[(shortest_paths_clipped==i)*~is_notch].std() for i in range(1,7)])
    P_by_shortest_path_N_av = np.array([P[(shortest_paths_clipped==i)*~is_notch].mean() for i in range(1,7)])
    P_by_shortest_path_N_std = np.array([P[(shortest_paths_clipped==i)*~is_notch].std() for i in range(1,7)])
    H_by_shortest_path_N_av = np.array([H[(shortest_paths_clipped==i)*~is_notch].mean() for i in range(1,7)])
    H_by_shortest_path_N_std = np.array([H[(shortest_paths_clipped==i)*~is_notch].std() for i in range(1,7)])

    cell_props_by_shortest_path_dict = dict(
        zip(["A_by_shortest_path_av_%d" % i for i in range(1,7)], A_by_shortest_path_av))
    cell_props_by_shortest_path_dict.update(
        dict(zip(["A_by_shortest_path_std_%d" % i for i in range(1,7)], A_by_shortest_path_std)))
    cell_props_by_shortest_path_dict.update(
        dict(zip(["P_by_shortest_path_av_%d" % i for i in range(1,7)], P_by_shortest_path_av)))
    cell_props_by_shortest_path_dict.update(
        dict(zip(["P_by_shortest_path_std_%d" % i for i in range(1,7)], P_by_shortest_path_std)))
    cell_props_by_shortest_path_dict.update(
        dict(zip(["H_by_shortest_path_av_%d" % i for i in range(1,7)], H_by_shortest_path_av)))
    cell_props_by_shortest_path_dict.update(dict(zip(["H_by_shortest_path_std_%d" % i for i in range(1,7)],
                 H_by_shortest_path_std)))

    cell_props_by_shortest_path_dict.update(dict(
        zip(["A_by_shortest_path_P_av_%d" % i for i in range(1,7)], A_by_shortest_path_P_av)))
    cell_props_by_shortest_path_dict.update(
        dict(zip(["A_by_shortest_path_P_std_%d" % i for i in range(1,7)], A_by_shortest_path_P_std)))
    cell_props_by_shortest_path_dict.update(
        dict(zip(["P_by_shortest_path_P_av_%d" % i for i in range(1,7)], P_by_shortest_path_P_av)))
    cell_props_by_shortest_path_dict.update(
        dict(zip(["P_by_shortest_path_P_std_%d" % i for i in range(1,7)], P_by_shortest_path_P_std)))
    cell_props_by_shortest_path_dict.update(
        dict(zip(["H_by_shortest_path_P_av_%d" % i for i in range(1,7)], H_by_shortest_path_P_av)))
    cell_props_by_shortest_path_dict.update(dict(zip(["H_by_shortest_path_P_std_%d" % i for i in range(1,7)],
                 H_by_shortest_path_P_std)))

    cell_props_by_shortest_path_dict.update(dict(
        zip(["A_by_shortest_path_N_av_%d" % i for i in range(1,7)], A_by_shortest_path_N_av)))
    cell_props_by_shortest_path_dict.update(
        dict(zip(["A_by_shortest_path_N_std_%d" % i for i in range(1,7)], A_by_shortest_path_N_std)))
    cell_props_by_shortest_path_dict.update(
        dict(zip(["P_by_shortest_path_N_av_%d" % i for i in range(1,7)], P_by_shortest_path_N_av)))
    cell_props_by_shortest_path_dict.update(
        dict(zip(["P_by_shortest_path_N_std_%d" % i for i in range(1,7)], P_by_shortest_path_N_std)))
    cell_props_by_shortest_path_dict.update(
        dict(zip(["H_by_shortest_path_N_av_%d" % i for i in range(1,7)], H_by_shortest_path_N_av)))
    cell_props_by_shortest_path_dict.update(dict(zip(["H_by_shortest_path_N_std_%d" % i for i in range(1,7)],
                 H_by_shortest_path_N_std)))

    A_by_n_cc_av = np.array([A[n_cc_by_cell_clipped == i].mean() for i in range(1, 9)])
    A_by_n_cc_std = np.array([A[n_cc_by_cell_clipped == i].std() for i in range(1, 9)])
    P_by_n_cc_av = np.array([P[n_cc_by_cell_clipped == i].mean() for i in range(1, 9)])
    P_by_n_cc_std = np.array([P[n_cc_by_cell_clipped == i].std() for i in range(1, 9)])
    H_by_n_cc_av = np.array([H[n_cc_by_cell_clipped == i].mean() for i in range(1, 9)])
    H_by_n_cc_std = np.array([H[n_cc_by_cell_clipped == i].std() for i in range(1, 9)])

    cell_props_by_n_cc_dict = dict(
        zip(["A_by_n_cc_av_%d" % i for i in range(1,9)], A_by_n_cc_av))
    cell_props_by_n_cc_dict.update(dict(
        zip(["A_by_n_cc_std_%d" % i for i in range(1, 9)], A_by_n_cc_std)))
    cell_props_by_n_cc_dict.update(dict(
        zip(["P_by_n_cc_av_%d" % i for i in range(1, 9)], P_by_n_cc_av)))
    cell_props_by_n_cc_dict.update(dict(
        zip(["P_by_n_cc_std_%d" % i for i in range(1, 9)], P_by_n_cc_std)))
    cell_props_by_n_cc_dict.update(dict(
        zip(["H_by_n_cc_av_%d" % i for i in range(1, 9)], H_by_n_cc_av)))
    cell_props_by_n_cc_dict.update(dict(
        zip(["H_by_n_cc_std_%d" % i for i in range(1, 9)], H_by_n_cc_std)))
    A_by_n_cc_P_av = np.array([A[(n_cc_by_cell_clipped == i) * is_notch].mean() for i in range(1, 9)])
    A_by_n_cc_P_std = np.array([A[(n_cc_by_cell_clipped == i) * is_notch].std() for i in range(1, 9)])
    P_by_n_cc_P_av = np.array([P[(n_cc_by_cell_clipped == i) * is_notch].mean() for i in range(1, 9)])
    P_by_n_cc_P_std = np.array([P[(n_cc_by_cell_clipped == i) * is_notch].std() for i in range(1, 9)])
    H_by_n_cc_P_av = np.array([H[(n_cc_by_cell_clipped == i) * is_notch].mean() for i in range(1, 9)])
    H_by_n_cc_P_std = np.array([H[(n_cc_by_cell_clipped == i) * is_notch].std() for i in range(1, 9)])

    cell_props_by_n_cc_dict.update(dict(
        zip(["A_by_n_cc_P_av_%d" % i for i in range(1,9)], A_by_n_cc_P_av)))
    cell_props_by_n_cc_dict.update(dict(
        zip(["A_by_n_cc_P_std_%d" % i for i in range(1, 9)], A_by_n_cc_P_std)))
    cell_props_by_n_cc_dict.update(dict(
        zip(["P_by_n_cc_P_av_%d" % i for i in range(1, 9)], P_by_n_cc_P_av)))
    cell_props_by_n_cc_dict.update(dict(
        zip(["P_by_n_cc_P_std_%d" % i for i in range(1, 9)], P_by_n_cc_P_std)))
    cell_props_by_n_cc_dict.update(dict(
        zip(["H_by_n_cc_P_av_%d" % i for i in range(1, 9)], H_by_n_cc_P_av)))
    cell_props_by_n_cc_dict.update(dict(
        zip(["H_by_n_cc_P_std_%d" % i for i in range(1, 9)], H_by_n_cc_P_std)))

    A_by_n_cc_N_av = np.array([A[(n_cc_by_cell_clipped == i) * ~is_notch].mean() for i in range(1, 9)])
    A_by_n_cc_N_std = np.array([A[(n_cc_by_cell_clipped == i) * ~is_notch].std() for i in range(1, 9)])
    P_by_n_cc_N_av = np.array([P[(n_cc_by_cell_clipped == i) * ~is_notch].mean() for i in range(1, 9)])
    P_by_n_cc_N_std = np.array([P[(n_cc_by_cell_clipped == i) * ~is_notch].std() for i in range(1, 9)])
    H_by_n_cc_N_av = np.array([H[(n_cc_by_cell_clipped == i) * ~is_notch].mean() for i in range(1, 9)])
    H_by_n_cc_N_std = np.array([H[(n_cc_by_cell_clipped == i) * ~is_notch].std() for i in range(1, 9)])



    cell_props_by_n_cc_dict.update(dict(
        zip(["A_by_n_cc_N_av_%d" % i for i in range(1,9)], A_by_n_cc_N_av)))
    cell_props_by_n_cc_dict.update(dict(
        zip(["A_by_n_cc_N_std_%d" % i for i in range(1, 9)], A_by_n_cc_N_std)))
    cell_props_by_n_cc_dict.update(dict(
        zip(["P_by_n_cc_N_av_%d" % i for i in range(1, 9)], P_by_n_cc_N_av)))
    cell_props_by_n_cc_dict.update(dict(
        zip(["P_by_n_cc_N_std_%d" % i for i in range(1, 9)], P_by_n_cc_N_std)))
    cell_props_by_n_cc_dict.update(dict(
        zip(["H_by_n_cc_N_av_%d" % i for i in range(1, 9)], H_by_n_cc_N_av)))
    cell_props_by_n_cc_dict.update(dict(
        zip(["H_by_n_cc_N_std_%d" % i for i in range(1, 9)], H_by_n_cc_N_std)))

    dicts = [cell_properties,z_by_positive_dict,n_notch_neighbours_count_dict,cell_props_by_n_notch_neighbours_dict,graph_props_dict,cell_props_by_shortest_path_dict,cell_props_by_n_cc_dict]
    master_dict = {}
    for dct in dicts:
        master_dict.update(dct)
    return master_dict


if __name__ == "__main__":

    N_param = 12
    N_iter = 12
    T_cortical_range = np.linspace(0,0.5,N_param)
    alpha_range = np.linspace(0,1,N_param)
    A0_range = np.linspace(2,7,N_param)
    p_notch_range = np.linspace(0.05,0.95,N_param)
    seed_index = np.arange(N_iter)

    TC,AL,A0,PN,S = np.array(np.meshgrid(T_cortical_range,alpha_range,A0_range,p_notch_range,seed_index,indexing="ij")).reshape(5,-1)
    S = S.astype(int)

    slurm_index = int(sys.argv[1])
    N_batch = 864
    n_batches = 288

    range_to_index = np.arange(slurm_index*N_batch,(slurm_index+1)*N_batch)

    TC_i, AL_i, A0_i, PN_i, S_i = TC[range_to_index],AL[range_to_index],A0[range_to_index],PN[range_to_index],S[range_to_index]
    mkdir("../results/export_dump/batch_%d"%slurm_index)
    file_name = "../results/export_dump/batch_%d"%slurm_index
    out_dicts = Parallel(n_jobs=-1)(delayed(run_simulation)(T_cortical,alpha,A0,p_notch,file_name,index) for (T_cortical,alpha,A0,p_notch,index) in zip(TC_i, AL_i, A0_i, PN_i, S_i))

    df_out = pd.DataFrame(out_dicts)
    df_out.to_csv("../results/statistics_dump/batch_%d.csv"%slurm_index)

    os.system("tar -zcvf -r ../results/export/batch_%d.tar.gz ../results/export_dump/batch_%d"%slurm_index)
    os.system("rm -R ../results/export_dump/batch_%d"%slurm_index)

    #mkdir("../results")
    #mkdir("../results/export")
    #mkdir("../results/export_dump")
    #mkdir("../results/statistics")
    #mkdir("../results/statistics_dump")

