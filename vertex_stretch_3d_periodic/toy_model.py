import sys
sys.dont_write_bytecode = True
import os
SCRIPT_DIR = "../../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

from vertex_stretch_3d_periodic.simulation import Simulation,plot_3d
from vertex_stretch_3d_periodic.mesh import assemble_scalar
from vertex_stretch_3d_periodic import tri_functions as trf

import numpy as np
import networkx as nx
from scipy import sparse
import pandas as pd
from copy import deepcopy
import pickle
import bz2
import os
from joblib import Parallel, delayed
import matplotlib
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import numba as nb

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

# T_cortical,alpha,A0,p_notch,file_name,seed = (0.3, 0.0, 6.454545454545455, 0.5, 'results', 3)


T_cortical,alpha,A0,p_notch,file_name,seed = (0.1909090909090909, 0.4545454545454546, 7.045454545454546, 0.3, 'results', 3)

mesh_options = {"L": 18.4, "A_init": 2., "V_init": 1., "init_noise": 1e-1, "eps": 0.002, "l_mult": 1.05,"seed":seed+2024}
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
              "mu_L": 0.5, ##beware i changed form 0.25
              "L_min": 2.,
              "max_L_grad":100}
simulation_options = {"dt": 0.01,
                  "tfin": 10,
                  "t_skip": 10}
sim = Simulation(simulation_options, tissue_options, mesh_options)

n_c = int(sim.t.mesh.mesh_props["n_c"])

n_iter = 30000
is_notchs = []
gt_clusters = np.zeros((n_iter, 2))
for i in tqdm(range(n_iter)):
    is_notch = np.zeros(n_c, dtype=bool)
    is_notch[:int(np.round(p_notch * n_c))] = True
    np.random.shuffle(is_notch)
    is_notchs.append(is_notch.copy())
    ##Calculate the adjacency matrix

    is_notch_int = is_notch.astype(int)
    is_notch_edges = is_notch_int[edges]

    adj_notch = sparse.coo_matrix((np.ones((is_notch_edges[:, 0] == is_notch_edges[:, 1]).sum(), dtype=int), (
        edges[(is_notch_edges[:, 0] == is_notch_edges[:, 1]), 0],
        edges[(is_notch_edges[:, 0] == is_notch_edges[:, 1]), 1])),
                                  shape=(int(mesh_props["n_c"]), int(mesh_props["n_c"])))

    n_cc, cc_labs = sparse.csgraph.connected_components(adj_notch)
    gt_cluster_pos = np.bincount(cc_labs[is_notch]).max()
    gt_cluster_neg = np.bincount(cc_labs[~is_notch]).max()
    gt_clusters[i] = gt_cluster_pos, gt_cluster_neg

opt_area_N = 1
opt_area_P = 2
V0 = 1

N_cell = 100
p_notch = 0.2



def get_E(A_P,A_N,p_notch,beta):
    H_P,H_N = V0/A_P, V0/A_N

    dH_P = beta*(1-p_notch)*(H_P-H_N)**2
    dH_N = beta*(p_notch)*(H_P-H_N)**2


    E = p_notch*(opt_area_P-A_P)**2 + (1-p_notch)*(opt_area_N-A_N)**2 + p_notch*dH_P + (1-p_notch)*dH_N
    return E

A_range = np.linspace(0.5,2.5,100)

A_P_range, A_N_range = np.meshgrid(A_range,A_range,indexing="ij")



plt.imshow(get_E(A_P_range,A_N_range,0.5,0.1),cmap=sns.color_palette("icefire",as_cmap=True))

plt.show()

beta = 3
p_notch_range = np.linspace(0.05,0.95,100)
A_opts = np.zeros((len(p_notch_range),2))
for i, p_notch in tqdm(enumerate(p_notch_range)):
    E = get_E(A_P_range, A_N_range, p_notch,beta)
    A_opts[i] = A_P_range[E==E.min()].mean(),A_N_range[E==E.min()].mean()


plt.plot(A_opts[:,0]*p_notch_range + A_opts[:,1]*(1-p_notch_range))
plt.show()

for j, p_notch in enumerate(np.linspace(0.1,0.7,10)):
    N_set = 6
    n_c = int(sim.t.mesh.mesh_props["n_c"])
    tri = np.array(sim.t.mesh.mesh_props["tri"])
    trip1 = np.roll(tri, 1, axis=1)
    mesh_props = sim.t.mesh.mesh_props
    edges = np.row_stack(
        [np.column_stack([mesh_props["tri"][:, i], mesh_props["tri"][:, (i + 1) % 3]]) for i in
         range(3)])
    is_notch = np.zeros(n_c, dtype=bool)
    is_notch[:int(np.round(p_notch * n_c))] = True
    np.random.shuffle(is_notch)
    ##Calculate the adjacency matrix

    is_notch_int = is_notch.astype(int)
    is_notch_edges = is_notch_int[edges]

    adj = sparse.coo_matrix((np.ones(len(edges), dtype=int), (edges[:, 0], edges[:, 1])),
                                               shape=(int(mesh_props["n_c"]), int(mesh_props["n_c"])))


    adj_notch = sparse.coo_matrix((np.ones((is_notch_edges[:, 0] == is_notch_edges[:, 1]).sum(), dtype=int), (
        edges[(is_notch_edges[:, 0] == is_notch_edges[:, 1]), 0],
        edges[(is_notch_edges[:, 0] == is_notch_edges[:, 1]), 1])),
                                  shape=(int(mesh_props["n_c"]), int(mesh_props["n_c"])))


    plt.hist(adj_notch.toarray().sum(axis=0),histtype="step",color=plt.cm.plasma(j/10),density=True,bins=3)




adj = adj.todense()


from jax import jit, jacrev,hessian
from scipy.optimize import minimize

pair_indices = adj.nonzero()
@jit
def get_E(A,T,A0,beta):

    H = A/V0
    P = 3.81*jnp.sqrt(jnp.abs(A))
    E = ((A-A0)**2).sum() + (T*P).sum() + beta*((H[pair_indices[0]]-H[pair_indices[1]])**2).sum()

    return E

jac = jit(jacrev(get_E))
hess = jit(hessian(get_E))


A_init = np.ones(169)*3 + np.random.normal(0,0.2,169)

p_notch_range = np.linspace(0.05,0.7,10)
beta_range = np.logspace(-3,4,10)

A_opt = np.zeros((10,10))
for i, p_notch in tqdm(enumerate(p_notch_range)):
    for j, beta in enumerate(beta_range):

        T_P = 0.030
        T_N = 0.30
        is_notch = np.zeros(169,dtype=bool)
        is_notch[:int(p_notch*169)] = True
        np.random.shuffle(is_notch)
        A0 = 3
        T = np.ones(169)*T_N
        T[is_notch] = T_P




        res = minimize(get_E,A_init,args=(T,A0,beta),jac=jac,hess=hess,method="Newton-CG")
        A_opt[i,j] = res.x.sum()







A_range = np.linspace(0.5,2.5,100)

A_P_range, A_N_range = np.meshgrid(A_range,A_range,indexing="ij")



plt.imshow(get_E(A_P_range,A_N_range,0.5,0.1),cmap=sns.color_palette("icefire",as_cmap=True))

plt.show()

beta = 3
p_notch_range = np.linspace(0.05,0.95,100)
A_opts = np.zeros((len(p_notch_range),2))
for i, p_notch in tqdm(enumerate(p_notch_range)):
    E = get_E(A_P_range, A_N_range, p_notch,beta)
    A_opts[i] = A_P_range[E==E.min()].mean(),A_N_range[E==E.min()].mean()


plt.plot(A_opts[:,0]*p_notch_range + A_opts[:,1]*(1-p_notch_range))
plt.show()



from jax import jit, jacrev, hessian
from scipy.optimize import minimize
@jit
def get_E(H, A_P,A_N,p_notch,beta,opt_area_P,opt_area_N):

    V_P, V_N = A_P*H, A_N*H

    E = p_notch*((opt_area_P-A_P)**2 + beta*(V_P-1)**2) + (1-p_notch)*((opt_area_N-A_N)**2 + beta*(V_N-1)**2)
    return E

@jit
def _get_E(X,p_notch,beta,opt_area_P,opt_area_N):
    return get_E(X[0], X[1],X[2],p_notch,beta,opt_area_P,opt_area_N)

jac = jit(jacrev(_get_E))
hess = jit(hessian(_get_E))

def get_opt_X(p_notch,beta,opt_area_P,opt_area_N):
    return minimize(_get_E,[0.1,1,2],args=(p_notch,beta,opt_area_P,opt_area_N),jac=jac,hess=hess,method="Newton-CG",options={"xtol":1e-7})


opt_area_P = 5.5
opt_area_N = 1.
beta_range = np.logspace(-2,3,10)
p_notch_range = np.linspace(0.05,0.95,10)
BB,PP = np.meshgrid(beta_range,p_notch_range,indexing="ij")

out = np.zeros((10,10,3))
for i, beta in tqdm(enumerate(beta_range)):
    for j, p_notch in enumerate(p_notch_range):

        out[i,j] = get_opt_X(p_notch,beta,opt_area_P,opt_area_N).x

A_opt = out[:,:,1]*PP + out[:,:,2]*(1-PP)


"""
Still this doesn't work. 

"""

