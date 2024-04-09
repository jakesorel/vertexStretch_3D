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
from jax import numpy as jnp
from jax import jacrev,hessian, jit
import triangle as tr



def hexagonal_lattice(_rows=3, _cols=3, noise=0.005, A=None,seed=1):
    """
    Assemble a hexagonal lattice
    :param rows: Number of rows in lattice
    :param cols: Number of columns in lattice
    :param noise: Noise added to cell locs (Gaussian SD)
    :return: points (nc x 2) cell coordinates.
    """
    if A is None:
        A = 1.
    _A = np.max(A)
    rows = int(np.round(_rows / np.sqrt(_A)))
    cols = int(np.round(_cols / np.sqrt(_A)))
    points = []
    np.random.seed(seed)
    for row in range(rows * 2):
        for col in range(cols):
            x = (col + (0.5 * (row % 2))) * np.sqrt(3)
            y = row * 0.5
            x += np.random.normal(0, noise)
            y += np.random.normal(0, noise)
            points.append((x, y))
    points = np.asarray(points)
    if A is not None:
        points = points * np.sqrt(2 * np.sqrt(3) / 3) * np.sqrt(_A)
    points = jnp.array(points)

    return points

from itertools import combinations


pts = hexagonal_lattice(4,4)
mid_pt = np.where(np.abs(np.linalg.norm(pts-pts.mean(axis=0),axis=1))==np.abs(np.linalg.norm(pts-pts.mean(axis=0),axis=1)).min())
tri = tr.triangulate({"vertices":np.array(pts)})["triangles"]
pts = pts[np.unique(tri[(tri == mid_pt[0][0]).any(axis=1)])]
notch_indices = [list(combinations(np.arange(7),i)) for i in range(8)]


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


opt_area_P = 2.
opt_area_N = 1.
beta_range = np.logspace(-2,3,50)
p_notch_range = np.arange(8)/7
BB,PP = np.meshgrid(beta_range,p_notch_range,indexing="ij")


out = np.zeros((50,len(p_notch_range),3))
for i, beta in tqdm(enumerate(beta_range)):
    for j, p_notch in enumerate(p_notch_range):

        out[i,j] = get_opt_X(p_notch,beta,opt_area_P,opt_area_N).x

A_opt = out[:,:,1]*PP + out[:,:,2]*(1-PP)

"""
pr(i in 7 notch positive) = p^i *(1-p)^(7-i) * 7 choose i
"""

from scipy.special import binom

def get_av_area(p_notch,beta_index):
    av_area = 0
    for i in range(8):

        if i > 0:
            j = i-1
            p_config = p_notch ** j * (1 - p_notch) ** (6 - j) * binom(6, j)
            P_area = out[beta_index,i,1]
            av_area += p_config*P_area*p_notch
        if i < 6:
            j = i
            p_config = p_notch ** j * (1 - p_notch) ** (6 - j) * binom(6, j)
            N_area = out[beta_index,i,2]
            av_area += (1-p_notch)*N_area*p_config
    return av_area

av_areas = np.zeros((20,len(out)))

for i, p_notch in enumerate(np.linspace(0,1,20)):
    for j in range(len(out)):
        av_areas[i,j] = get_av_area(p_notch,j)




av_areas = [get_av_area(p_notch) for p_notch in np.linspace(0,1,20)]



