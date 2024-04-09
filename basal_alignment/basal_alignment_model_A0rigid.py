"""
The model:

In a box of shape L*lx, L*ly where L is a free parameter,
degrees of freedom:
- L
- vertex positions in (x,y) in central positions.
- Plane vector in all cells
- Under constraint that volume must equal 1.


"""

import numpy as np
import jax.numpy as jnp
from jax import jit,vmap,jacrev,hessian
from functools import partial
import jax
from scipy import sparse
import triangle as tr
from scipy.optimize import minimize
from itertools import combinations
import networkx as nx
import networkx.algorithms.isomorphism as iso
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import optax
from scipy.special import binom


jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
import matplotlib.pyplot as plt


class MakeHexagonalMesh:
    def __init__(self,mesh_params):
        self.mesh_params = mesh_params
        self.mesh_props = {}
        self.initialize()

    def initialize(self):
        self.make_hexagonal_block()
        self.initialize_planes()

    def make_hexagonal_block(self):
        _x = hexagonal_lattice(7,7,self.mesh_params["mesh_noise"],seed = self.mesh_params["mesh_seed"])
        _x -= _x.mean()
        d_centre = np.linalg.norm(_x,axis=-1)
        mid_cell = np.argmin(d_centre)
        _tri = tr.triangulate({"vertices":np.array(_x)})["triangles"]
        neighbouring_cells = np.unique(_tri[(_tri==mid_cell).any(axis=1)])
        neighbours_in_tri = np.zeros_like(_tri,dtype=bool)
        for c in neighbouring_cells:
            neighbours_in_tri += _tri == c
        central_tris = _tri[neighbours_in_tri.all(axis=1)]
        two_in_tris = _tri[neighbours_in_tri.sum(axis=1)==2]
        one_in_tris = _tri[neighbours_in_tri.sum(axis=1)==1]

        all_tris = np.row_stack((central_tris,two_in_tris,one_in_tris))
        all_cells = np.unique(all_tris)
        all_cells = [mid_cell] + list(set(neighbouring_cells).difference(set([mid_cell]))) + list(set(all_cells).difference(list(neighbouring_cells)))
        cell_dict = np.zeros(len(_x),dtype=int)
        for i, c in enumerate(all_cells):
            cell_dict[c] = i


        tri = cell_dict[all_tris]
        r = _x[np.array(all_cells)]
        self.mesh_props["r"] = r
        self.mesh_props["x"] = circumcenter(self.mesh_props["r"][tri])
        self.mesh_props["tri"] = tri
        self.mesh_props["neigh"] = get_neighbours(tri)
        self.mesh_props["k2s"] = get_k2(tri,self.mesh_props["neigh"])
        self.mesh_props["L"] = 1.

    def initialize_planes(self):
        self.mesh_props["n"] = np.zeros((len(self.mesh_props["r"]),2))



def get_k2(tri, neigh):
    """
    To determine whether a given neighbouring pair of triangles needs to be re-triangulated, one considers the sum of
    the pair angles of the triangles associated with the cell centroids that are **not** themselves associated with the
    adjoining edge. I.e. these are the **opposite** angles.

    Given one cell centroid/angle in a given triangulation, k2 defines the column index of the cell centroid/angle in the **opposite** triangle

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: Neighbourhood matrix (n_v x 3) np.int32 array
    :return:
    """
    three = np.array([0, 1, 2])
    nv = tri.shape[0]
    k2s = np.empty((nv, 3), dtype=np.int32)
    for i in range(nv):
        for k in range(3):
            neighbour = neigh[i, k]
            k2 = ((neigh[neighbour] == i) * three).sum()
            k2s[i, k] = k2
    return k2s



def get_neighbours(tri):
    """
    Given a triangulation, find the neighbouring triangles of each triangle.

    By convention, the column i in the output -- neigh -- corresponds to the triangle that is opposite the cell i in that triangle.

    Can supply neigh, meaning the algorithm only fills in gaps (-1 entries)

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: neighbourhood matrix to update {Optional}
    :return: (n_v x 3) np.int32 array, storing the three neighbouring triangles. Values correspond to the row numbers of tri
    """
    n_v = tri.shape[0]
    neigh = np.ones_like(tri, dtype=np.int32) * -1
    tri_compare = np.concatenate((tri.T, tri.T)).T.reshape((-1, 3, 2))
    for j in range(n_v):
        tri_sample_flip = np.flip(tri[j])
        tri_i = np.concatenate((tri_sample_flip, tri_sample_flip)).reshape(3, 2)
        for k in range(3):
            if neigh[j, k] == -1:
                mask = (tri_compare[:, :, 0] == tri_i[k, 0]) * (tri_compare[:, :, 1] == tri_i[k, 1])
                if mask.sum()>0:
                    neighb, l = np.nonzero(mask)
                    neighb, l = neighb[0], l[0]
                    neigh[j, k] = neighb
                    neigh[neighb, np.mod(2 - l, 3)] = j
    return neigh.astype(np.int32)




@partial(jit, static_argnums=(1,))
def get_geometry(mesh_props,nc):
    mesh_props["X"] = mesh_props["x"]*mesh_props["L"]
    mesh_props["_x"] = jnp.row_stack((mesh_props["x"],jnp.zeros((1,2))))
    mesh_props["_X"] = jnp.row_stack((mesh_props["X"],jnp.zeros((1,2))))
    mesh_props["x_"] = jnp.expand_dims(mesh_props["x"],axis=1)

    mesh_props["X_"] = jnp.expand_dims(mesh_props["X"],axis=1)
    mesh_props["neigh_p1"] = jnp.roll(mesh_props["neigh"], -1, axis=1)
    mesh_props["k2s_p1"] = jnp.roll(mesh_props["k2s"], -1, axis=1)


    # mesh_props["v_p1"] = mesh_props["v"][mesh_props["neigh_p1"],(mesh_props["k2s_p1"]+1)%3]



    mesh_props["X_p1"] = mesh_props["_X"][mesh_props["neigh_p1"]]
    mesh_props["tn"] = mesh_props["n"][mesh_props["tri"]]

    mesh_props["disp_t"] = mesh_props["X_"]  - mesh_props["X_p1"]
    mesh_props["lt"] = jnp.linalg.norm(mesh_props["disp_t"], axis=-1)
    mesh_props["P"] = assemble_scalar(mesh_props["lt"],mesh_props["tri"],nc)
    mesh_props = get_A(mesh_props,nc)
    mesh_props = get_COM(mesh_props, nc)
    mesh_props = vol_slanted_hexagon(mesh_props,nc)
    mesh_props = get_basal_values(mesh_props)
    mesh_props = get_z_squared_displacement(mesh_props)
    return mesh_props


@partial(jit, static_argnums=(1,))
def get_A(mesh_props, nc):
    mesh_props["tri_A"] = 0.5 * jnp.cross(mesh_props["X_"], mesh_props["X_p1"], axis=-1)
    mesh_props["A"] = assemble_scalar(mesh_props["tri_A"], mesh_props["tri"], nc)
    mesh_props["tA"] = mesh_props["A"][mesh_props["tri"]]
    return mesh_props

@partial(jit, static_argnums=(1,))
def get_COM(mesh_props, nc): #revised
    t_r = (mesh_props["X_"] + mesh_props["X_p1"]) * jnp.expand_dims(
        jnp.cross(mesh_props["X_"], mesh_props["X_p1"], axis=-1), 2)
    # mesh_props["n_v"] = assemble_scalar(jnp.ones_like(mesh_props["X"]), mesh_props["tri"], nc)
    r = jnp.column_stack(
        [assemble_scalar(t_r[..., 0], mesh_props["tri"], nc),
         assemble_scalar(t_r[..., 1], mesh_props["tri"], nc)]) / jnp.expand_dims(6 * mesh_props["A"] + 1e-17,
                                                                                 axis=-1)
    mesh_props["r"] = r
    mesh_props["tr"] = r[mesh_props["tri"]]
    return mesh_props




@partial(jit, static_argnums=(2,))
def assemble_scalar(tval, tri, nc):
    val = jnp.bincount(tri.ravel() + 1, weights=tval.ravel(), length=nc + 1)
    val = val[1:]
    return val




@jit
def scalar_triple(a,b,c,d):
    return jnp.dot((a-d),jnp.cross(b-d,c-d))

@jit
def slanted_prism_volume(X):
    X0 = X.copy()
    X0 = X0.at[:,2].set(0)

    tet1 = scalar_triple(X[0],X[1],X[2],X0[2])
    tet2 = scalar_triple(X[0],X0[0],X[1],X0[2])
    tet3 = scalar_triple(X0[0],X0[1],X[1],X0[2])
    return (tet1+tet2+tet3)/6


@partial(jit, static_argnums=(1,))
def vol_slanted_hexagon(mesh_props,nc):
    mesh_props["z0"] = jnp.sum((mesh_props["X_"]-mesh_props["tr"])*mesh_props["tn"],axis=-1)
    x_triples = jnp.column_stack([mesh_props["X_"]*jnp.ones((mesh_props["X_"].shape[0],3,2)),mesh_props["X_p1"],mesh_props["tr"]]).reshape(-1,3,3,2).transpose(0,2,1,3)
    mesh_props["z0_"] = jnp.expand_dims(mesh_props["z0"],axis=(2))


    Xz_triples = jnp.zeros((mesh_props["X_"].shape[0],3,3,3))
    Xz_triples = Xz_triples.at[:,:,:,:2].set(x_triples)
    Xz_triples = Xz_triples.at[:, :, :, 2].set(mesh_props["z0_"])


    mesh_props["tri_V"] = jnp.column_stack([vmap(slanted_prism_volume)(Xz_triples[:,i]) for i in range(3)])
    mesh_props["V"] = assemble_scalar(mesh_props["tri_V"], mesh_props["tri"], nc)

    return mesh_props



@jit
def get_basal_values(mesh_props):
    mesh_props["d"] = (mesh_props["V0"]-mesh_props["V"])/mesh_props["A"]
    mesh_props["td"] = mesh_props["d"][mesh_props["tri"]]
    mesh_props["z"] = mesh_props["td"] + mesh_props["z0"]
    mesh_props["_z"] = jnp.row_stack([mesh_props["z"],jnp.ones((1,3))])
    mesh_props["z_p1"] = mesh_props["_z"][mesh_props["neigh_p1"],(mesh_props["k2s_p1"]+1)%3]
    return mesh_props

@partial(jit, static_argnums=(1,))
def get_z_squared_displacement(mesh_props):
    """
    Consider for a pair of neighbouring vertices, the corresponding 4 values of z making two lines.
    Consider an infinite set of springs along this line.

    int (∆z)^2(x) dx

    Not 100% confident on the indexing here.
    """
    z1 = mesh_props["z"]
    z1_p1 = mesh_props["z_p1"]
    z2 = jnp.roll(mesh_props["z"],1,axis=1)
    z2_p1 = jnp.roll(mesh_props["z_p1"],1,axis=1)
    dz = z2-z1
    dzp1 = z2_p1 - z1_p1
    integral = mesh_props["lt"]/3 * (dz**2 + (dz)*(dzp1) + dzp1**2)
    mesh_props["int_t_dz_sq"] = integral
    # mesh_props["int_dz_sq"] = assemble_scalar(mesh_props["int_t_dz_sq"],mesh_props["tri"],nc)
    return mesh_props



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


def circumcenter(C):
    """
    Find the circumcentre (i.e. vertex position) of each triangle in the triangulation.

    :param C: Cell centroids for each triangle in triangulation (n_c x 3 x 2) np.float32 array
    :param L: Domain size (np.float32)
    :return: Circumcentres/vertex-positions (n_v x 2) np.float32 array
    """
    ri, rj, rk = C.transpose(1, 2, 0)
    ax, ay = ri
    bx, by = rj
    cx, cy = rk
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
    vs = np.empty((ax.size, 2), dtype=np.float32)
    vs[:, 0], vs[:, 1] = ux, uy
    return vs

@jit
def get_E_A(A,A0):
    return A*(1.0*(A>A0) + (2*A/A0 - 1)*(A<=A0))


@partial(jit, static_argnums=(3))
def get_E(Y,tissue_params,mesh_props,nc):
    """
    gamma external is weighted average of gammaP and gammaN by p_notch
    gamma external is attributed to all of the non-central cells (>=7)

    """
    # x_dof = Y[:12].reshape(6,2)
    x_dof = Y[:24].reshape(12,2)

    n_dof = Y[24:24+14].reshape(7,2)
    L = jnp.abs(Y[-1])
    mesh_props["x"] = mesh_props["x"].at[:12].set(x_dof)
    mesh_props["L"] = L
    mesh_props["n"] = mesh_props["n"].at[:7].set(n_dof)
    mesh_props = get_geometry(mesh_props,nc)
    E_A = tissue_params["kappa_A"]*get_E_A(mesh_props["A"],tissue_params["A0"])
    E_A = E_A[:7].sum()
    #
    # tgamma = tissue_params["gamma"][mesh_props["tri"]] ##ensure that gamma for all boundary cells is p_notch related.
    # tgamma_reciprocal = (tgamma + jnp.roll(tgamma,1,axis=1))
    # E_gamma = ((mesh_props["lt"]*tgamma_reciprocal)[mesh_props["tri_index_by_edge"][:,0],mesh_props["tri_index_by_edge"][:,1]]).sum()

    E_P = tissue_params["kappa_P"]*(mesh_props["P"]-tissue_params["P0"])**2
    E_P = E_P[:7].sum()


    both_in_mask = (mesh_props["tri"]<=6)*(jnp.roll(mesh_props["tri"],1,axis=1)<=6)
    E_int_z = tissue_params["Gamma"]*(mesh_props["int_t_dz_sq"]*both_in_mask).sum()/L**2

    return E_A + E_P + E_int_z + tissue_params["T_ex"]*mesh_props["A"][:7].sum()


jac = jit(jacrev(get_E),static_argnums=(3,))

def cost_grad(Y,tissue_params,mesh_props,nc):
    return np.array(jac(Y,tissue_params,mesh_props,nc))


hess = jit(hessian(get_E),static_argnums=(3,))

mesh_params = {"mesh_noise":1e-4,"mesh_seed":1}
minimizer_params = {"start_learning_rate":0.01,"maxiter":2000}

self = MakeHexagonalMesh(mesh_params)

mesh_props = self.mesh_props
mesh_props["V0"] = 1.
mesh_props["central_tri_index"] = jnp.nonzero((mesh_props["tri"]<7)*(jnp.roll(mesh_props["tri"],1,axis=1)<7))
nv = 24
nc = np.max(mesh_props["tri"])+1

n_positive = 3
# A0[positive_indices] = A0_pos
pos_gamma_mult = 0.0
positive_indices = []
gamma = np.ones(nc)*0.0257
gamma[positive_indices] *= pos_gamma_mult
p_notch = 0.
gamma[7:] = p_notch*gamma.min() + (1-p_notch)*gamma.max()
Gamma = 0.1
tissue_params = {"kappa_A": 0.02,
                 "A0": 5.2,
                 "P0":9.67,
                 "kappa_P":0.002,
                 "T_ex":-0.03,
                 "gamma": gamma,
                 "Gamma": Gamma}
n_dof = np.zeros((7,2))
x_dof = mesh_props["x"][:12]
L = 1.



edges = np.row_stack([np.roll(mesh_props["tri"],i,axis=1)[:,:2] for i in range(3)])

adj = sparse.coo_matrix((np.ones(len(edges),dtype=bool),(edges[:,0],edges[:,1])))
adj = adj + adj.T
edges_sorted = np.array(adj[:7].nonzero()).T
edges_sorted = edges_sorted[edges_sorted[:,1]>edges_sorted[:,0]]
tri_index_by_edge = np.zeros((len(edges_sorted),2),dtype=int)
for i, edge in enumerate(edges_sorted):
    mask_p1 = (mesh_props["tri"]==edge[0])*(np.roll(mesh_props["tri"],1,axis=1)==edge[1])
    tri_index_by_edge[i,:2] = np.array(np.nonzero(mask_p1))[:,0]

mesh_props["tri_index_by_edge"] = tri_index_by_edge


mesh_props["n"] = jnp.array(mesh_props["n"])
mesh_props["x"] = jnp.array(mesh_props["x"])
mesh_props["x"] = mesh_props["x"].at[:12].set(x_dof)
mesh_props["L"] = L
mesh_props["n"] = mesh_props["n"].at[:7].set(n_dof)
Y = np.concatenate((x_dof.ravel(),n_dof.ravel(),(1.,)))

E = get_E(Y,tissue_params,mesh_props,nc)

# j = jac(Y,tissue_params,mesh_props,nc)
# hess(Y,tissue_params,mesh_props,nc)

# res = minimize(get_E,Y,jac=jac,hess=hess,args=(tissue_params,mesh_props,nc),method="Newton-CG")
params = jnp.array(Y)
start_learning_rate = minimizer_params["start_learning_rate"]
optimizer = optax.adam(start_learning_rate)

opt_state = optimizer.init(params)
args = (tissue_params, mesh_props, nc)

iterator = tqdm(range(minimizer_params["maxiter"]))
# A simple update loop.
for i in iterator:
    grads = cost_grad(params, *args)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
print("iteration ", i, "E = ", get_E(params, *args))

# Y = res.x
Y = params
x_dof = Y[:24].reshape(12, 2)
n_dof = Y[24:24 + 14].reshape(7, 2)
L = jnp.abs(Y[-1])

mesh_props["n"] = jnp.array(mesh_props["n"])
mesh_props["x"] = jnp.array(mesh_props["x"])
mesh_props["x"] = mesh_props["x"].at[:12].set(x_dof)
mesh_props["L"] = L
mesh_props["n"] = mesh_props["n"].at[:7].set(n_dof)
mesh_props = get_geometry(mesh_props, nc)

fig, ax = plt.subplots()
ax.scatter(*mesh_props["X"].T,c=mesh_props["z"].mean(axis=1))
# ax.scatter(*mesh_props["r"][:7].T)
# ax.quiver(mesh_props["r"][:7,0],mesh_props["r"][:7,1],mesh_props["n"][:7,0],mesh_props["n"][:7,1],color=plt.cm.plasma(A0[:7]/6))
fig.show()





def identify_configurations(mesh_props):
    edges = [tuple(edge) for edge in np.row_stack([np.roll(mesh_props["tri"][:6],i,axis=1)[:,:2] for i in range(3)])]

    G = nx.Graph()
    G.add_edges_from(edges)
    graph_dict = {}
    n_options = 0
    _is_notch = np.zeros((7),dtype=bool)
    graph_dict = {}
    for j in range(8):
        graph_dict[j] = []
        combos = combinations(np.arange(7),j)
        for c in combos:
            is_notch = _is_notch.copy()
            is_notch[[c]] = True
            _G = G.copy()
            nx.set_node_attributes(_G, dict(zip(np.arange(7),is_notch)), 'label')
            g = {}
            g["G"] = _G
            g["is_notch"] = is_notch
            graph_dict[j].append(g)
            n_options +=1


    nm = iso.categorical_node_match("label", True)
    N = 0
    non_isomorphic_graphs = {}
    q = 0
    for j in range(8):
        if (j == 0)or(j==7):
            non_isomorphic_graphs[q] = graph_dict[j][0]
            non_isomorphic_graphs[q]["n_pos"] = j
            non_isomorphic_graphs[q]["n_instance"] = 1
            q += 1
            N += 1

        else:
            g_list = graph_dict[j]
            combos = list(combinations(np.arange(len(g_list)),2))
            isomorphic = np.zeros(len(combos),dtype=bool)
            for k, (g1_i,g2_i) in enumerate(combos):
                g1,g2 = g_list[g1_i],g_list[g2_i]
                isomorphic[k] = nx.is_isomorphic(g1["G"], g2["G"], node_match=nm)
            M = sparse.coo_matrix((isomorphic,(np.array(combos)[:,0],np.array(combos)[:,1])),shape=(len(g_list),len(g_list)))
            M = M + M.T + sparse.coo_matrix(([True]*len(g_list),(np.arange(len(g_list)),np.arange(len(g_list)))))
            n_unique_isomorphics,isomorphic_labels = sparse.csgraph.connected_components(M)
            N += n_unique_isomorphics
            for i in range(n_unique_isomorphics):
                ii = np.argmax(isomorphic_labels==i)
                non_isomorphic_graphs[q] = g_list[ii]
                non_isomorphic_graphs[q]["n_pos"] = j
                non_isomorphic_graphs[q]["n_instance"] = np.sum(isomorphic_labels==i)
                q+=1
    return non_isomorphic_graphs

from copy import deepcopy

def get_A_tot(positive_indices,gamma_val,Gamma,_mesh_props,minimizer_params,_tissue_params):
    mesh_props = deepcopy(_mesh_props)
    tissue_params = deepcopy(_tissue_params)
    # A0[positive_indices] = A0_pos
    pos_gamma_mult = 0.0
    gamma = np.ones(nc) * gamma_val
    gamma[positive_indices] *= pos_gamma_mult
    p_notch = 0.
    gamma[7:] = p_notch * gamma.min() + (1 - p_notch) * gamma.max()
    p0 = tissue_params["P0"]
    tissue_params["P0"] = np.zeros(nc)
    tissue_params["P0"][positive_indices] = p0


    tissue_params["Gamma"] = Gamma


    n_dof = np.zeros((7, 2))
    x_dof = mesh_props["x"][:12]
    L = 1.

    edges = np.row_stack([np.roll(mesh_props["tri"], i, axis=1)[:, :2] for i in range(3)])

    adj = sparse.coo_matrix((np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])))
    adj = adj + adj.T
    edges_sorted = np.array(adj[:7].nonzero()).T
    edges_sorted = edges_sorted[edges_sorted[:, 1] > edges_sorted[:, 0]]
    tri_index_by_edge = np.zeros((len(edges_sorted), 2), dtype=int)
    for i, edge in enumerate(edges_sorted):
        mask_p1 = (mesh_props["tri"] == edge[0]) * (np.roll(mesh_props["tri"], 1, axis=1) == edge[1])
        tri_index_by_edge[i, :2] = np.array(np.nonzero(mask_p1))[:, 0]

    mesh_props["tri_index_by_edge"] = tri_index_by_edge

    mesh_props["n"] = jnp.array(mesh_props["n"])
    mesh_props["x"] = jnp.array(mesh_props["x"])
    mesh_props["x"] = mesh_props["x"].at[:12].set(x_dof)
    mesh_props["L"] = L
    mesh_props["n"] = mesh_props["n"].at[:7].set(n_dof)
    Y = np.concatenate((x_dof.ravel(), n_dof.ravel(), (1.,)))

    E = get_E(Y, tissue_params, mesh_props, nc)

    # j = jac(Y,tissue_params,mesh_props,nc)
    # hess(Y,tissue_params,mesh_props,nc)

    # res = minimize(get_E,Y,jac=jac,hess=hess,args=(tissue_params,mesh_props,nc),method="Newton-CG")
    params = jnp.array(Y)
    start_learning_rate = minimizer_params["start_learning_rate"]
    optimizer = optax.adam(start_learning_rate)

    opt_state = optimizer.init(params)
    args = (tissue_params, mesh_props, nc)

    iterator = tqdm(range(minimizer_params["maxiter"]))
    # A simple update loop.
    for i in iterator:
        grads = cost_grad(params, *args)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    print("iteration ", i, "E = ", get_E(params, *args))

    # Y = res.x
    Y = params
    x_dof = Y[:24].reshape(12, 2)
    n_dof = Y[24:24 + 14].reshape(7, 2)
    L = jnp.abs(Y[-1])

    mesh_props["n"] = jnp.array(mesh_props["n"])
    mesh_props["x"] = jnp.array(mesh_props["x"])
    mesh_props["x"] = mesh_props["x"].at[:12].set(x_dof)
    mesh_props["L"] = L
    mesh_props["n"] = mesh_props["n"].at[:7].set(n_dof)
    mesh_props = get_geometry(mesh_props, nc)

    return mesh_props["A"][:7].sum(),L



mesh_params = {"mesh_noise": 1e-4, "mesh_seed": 1}
self = MakeHexagonalMesh(mesh_params)

mesh_props = self.mesh_props
mesh_props["V0"] = 1.
mesh_props["central_tri_index"] = jnp.nonzero(
    (mesh_props["tri"] < 7) * (jnp.roll(mesh_props["tri"], 1, axis=1) < 7))
nv = 24
nc = np.max(mesh_props["tri"]) + 1




non_isomorphic_graphs = identify_configurations(mesh_props)
is_notchs = [g["is_notch"] for g in non_isomorphic_graphs.values()]
count = [g["n_pos"] for g in non_isomorphic_graphs.values()]
freq = [g["n_instance"] for g in non_isomorphic_graphs.values()]

minimizer_params = {"start_learning_rate":0.01,"maxiter":500}


tissue_params = {"kappa_A": 0.02,
                 "A0": 1.,
                 "P0": 9.67,
                 "kappa_P": 10**-2.7,
                 "T_ex": -0.03,
                 "gamma":0}

gamma = 0.02
# A_vals,L_vals = np.array([get_A_tot(np.nonzero(is_notch)[0],gamma,0.1,mesh_props,minimizer_params,tissue_params) for is_notch in tqdm(is_notchs)]).T
#
# plt.scatter(count,A_vals)
# plt.show()

_A,_L = get_A_tot(np.nonzero(is_notchs[0])[0],gamma,0.,mesh_props,minimizer_params,tissue_params)
print(_A,_L)

Gamma_range = 10.0**(np.array([-6,2]))
A_vals,L_vals = np.zeros((len(Gamma_range),26)),np.zeros((len(Gamma_range),26))
for i, Gamma in enumerate(Gamma_range):
    A_vals[i],L_vals[i] = np.array([get_A_tot(np.nonzero(is_notch)[0],gamma,Gamma,mesh_props,minimizer_params,tissue_params) for is_notch in tqdm(is_notchs)]).T

fig, ax = plt.subplots()
for i, Gamma in enumerate(Gamma_range[1:]):
    ax.scatter(count,A_vals[i],color=plt.cm.plasma(i/len(Gamma_range)))
ax.set(xlabel="n_positive",ylabel="A_tot")
fig.show()



df_dict = {"count":count,"freq":freq}

for i, Gamma in enumerate(Gamma_range):
    df_dict["A_%d"%i] = A_vals[i]

df = pd.DataFrame(df_dict)
fig, ax = plt.subplots()
for i in range(len(Gamma_range)):
    sns.lineplot(data=df,x="count",y="A_%d"%i,color=plt.cm.plasma(i/len(Gamma_range)),label="Gamma=%.2f"%Gamma_range[i])

fig.show()

def get_A_av(A_vals):

    p_notch_range = np.linspace(0,1)
    tot_freq = np.bincount(count,freq)
    A_avs = np.zeros_like(p_notch_range)
    for j, p_notch in enumerate(p_notch_range):
        A_av = 0
        for c,f,A in zip(count,freq,A_vals):
            p_count = binom(7,c)*(p_notch)**c * (1-p_notch)**(7-c)
            p_config = f/tot_freq[c]
            p = p_count*p_config
            A_av += p*A
        A_avs[j] = A_av
    return A_avs

fig, ax = plt.subplots()
for i in range(len(Gamma_range)):
    ax.plot(np.linspace(0,1),get_A_av(A_vals[i]),color=plt.cm.plasma(i/len(Gamma_range)))
fig.show()






#########################
## energy landscape with L
##########################



@partial(jit, static_argnums=(4,))
def get_E_L(Y,L,tissue_params,mesh_props,nc):
    x_dof = Y[:24].reshape(12,2)

    n_dof = Y[24:24+14].reshape(7,2)
    mesh_props["x"] = mesh_props["x"].at[:12].set(x_dof)
    mesh_props["L"] = L
    mesh_props["n"] = mesh_props["n"].at[:7].set(n_dof)
    mesh_props = get_geometry(mesh_props,nc)
    E_A = tissue_params["kappa_A"]*get_E_A(mesh_props["A"],tissue_params["A0"])
    E_A = E_A[:7].sum()

    tgamma = tissue_params["gamma"][mesh_props["tri"]] ##ensure that gamma for all boundary cells is p_notch related.
    tgamma_reciprocal = (tgamma + jnp.roll(tgamma,1,axis=1))
    E_gamma = ((mesh_props["lt"]*tgamma_reciprocal)[mesh_props["tri_index_by_edge"][:,0],mesh_props["tri_index_by_edge"][:,1]]).sum()

    E_P = tissue_params["kappa_P"]*(mesh_props["P"]-tissue_params["P0"])**2
    E_P = E_P[:7].sum()


    both_in_mask = (mesh_props["tri"]<=6)*(jnp.roll(mesh_props["tri"],1,axis=1)<=6)
    E_int_z = tissue_params["Gamma"]*(mesh_props["int_t_dz_sq"]*both_in_mask).sum()/L**2

    return E_A + E_P + E_gamma + E_int_z + tissue_params["T_ex"]*mesh_props["A"][:7].sum()
  # return E_i[:7].sum() + tissue_params["Gamma"]*mesh_props["int_t_dz_sq"][mesh_props["central_tri_index"]].sum() ##penalise only central cells.




jac_L = jit(jacrev(get_E_L),static_argnums=(4,))
hess_L = jit(hessian(get_E_L),static_argnums=(4,))

def cost_grad_L(Y,L,tissue_params,mesh_props,nc):
    return np.array(jac_L(Y,L,tissue_params,mesh_props,nc))

mesh_params = {"mesh_noise":1e-4,"mesh_seed":1}
self = MakeHexagonalMesh(mesh_params)

mesh_props = self.mesh_props
mesh_props["V0"] = 1.
mesh_props["central_tri_index"] = jnp.nonzero((mesh_props["tri"]<7)*(jnp.roll(mesh_props["tri"],1,axis=1)<7))
nv = 24
nc = np.max(mesh_props["tri"])+1



tissue_params = {"kappa_A": 0.02,
                 "A0": 5.2,
                 "P0": 9.67,
                 "kappa_P": 10**-2.7,
                 "T_ex": -0.03}

gamma_val = 0.0257*1.5
Gamma = 0.
gamma = np.ones(nc) * gamma_val
gamma[positive_indices] *= pos_gamma_mult
p_notch = 0.
gamma[7:] = p_notch * gamma.min() + (1 - p_notch) * gamma.max()
tissue_params["gamma"] = gamma
tissue_params["Gamma"] = Gamma

# A0[positive_indices] = A0_pos
positive_indices = [0,1,2]

n_dof = np.zeros((7,2))
x_dof = mesh_props["x"][:12]
L = 0.1

mesh_props["n"] = jnp.array(mesh_props["n"])
mesh_props["x"] = jnp.array(mesh_props["x"])
mesh_props["x"] = mesh_props["x"].at[:12].set(x_dof)
mesh_props["L"] = L
mesh_props["n"] = mesh_props["n"].at[:7].set(n_dof)
Y = np.concatenate((x_dof.ravel(),n_dof.ravel()))
Y0 = Y.copy()
x0 = mesh_props["x"][:12].copy()
# j = jac(Y,tissue_params,mesh_props,nc)
# hess(Y,tissue_params,mesh_props,nc)


edges = np.row_stack([np.roll(mesh_props["tri"],i,axis=1)[:,:2] for i in range(3)])

adj = sparse.coo_matrix((np.ones(len(edges),dtype=bool),(edges[:,0],edges[:,1])))
adj = adj + adj.T
edges_sorted = np.array(adj[:7].nonzero()).T
edges_sorted = edges_sorted[edges_sorted[:,1]>edges_sorted[:,0]]
tri_index_by_edge = np.zeros((len(edges_sorted),2),dtype=int)
for i, edge in enumerate(edges_sorted):
    mask_p1 = (mesh_props["tri"]==edge[0])*(np.roll(mesh_props["tri"],1,axis=1)==edge[1])
    tri_index_by_edge[i,:2] = np.array(np.nonzero(mask_p1))[:,0]

mesh_props["tri_index_by_edge"] = tri_index_by_edge


res = minimize(get_E_L,Y,jac=jac_L,hess=hess_L,args=(L,tissue_params,mesh_props,nc),method="Newton-CG")

Y = res.x
x_dof = Y[:24].reshape(6, 2)
n_dof = Y[24:24 + 14].reshape(7, 2)
# L = jnp.abs(Y[-1])

mesh_props["n"] = jnp.array(mesh_props["n"])
mesh_props["x"] = jnp.array(mesh_props["x"])
mesh_props["x"] = mesh_props["x"].at[:12].set(x_dof)
mesh_props["L"] = L
mesh_props["n"] = mesh_props["n"].at[:7].set(n_dof)
mesh_props = get_geometry(mesh_props, nc)

plt.scatter(*mesh_props["x"].T)
plt.scatter(*mesh_props["x"][(mesh_props["tri"]==0).any(axis=1)].T)
plt.show()


def get_E_L_eq(L,positive_indices,gamma_val,Gamma,_mesh_props,minimizer_params,tissue_params):
    mesh_props = deepcopy(_mesh_props)
    gamma = np.ones(nc) * gamma_val
    gamma[positive_indices] *= 0
    p_notch = 0.
    gamma[7:] = p_notch * gamma.min() + (1 - p_notch) * gamma.max()
    tissue_params["gamma"] = gamma
    tissue_params["Gamma"] = Gamma
    n_dof = np.zeros((7,2))
    x_dof = mesh_props["x"][:12]
    Y = np.concatenate((x_dof.ravel(),n_dof.ravel()))

    # res = minimize(get_E_L, Y, jac=jac_L, hess=hess_L, args=(L,tissue_params, mesh_props, nc), method="Newton-CG",options={"xtol":1e-6})
    start_learning_rate = minimizer_params["start_learning_rate"]
    optimizer = optax.adam(start_learning_rate)

    # Initialize parameters of the model + optimizer.
    params = jnp.array(Y)
    opt_state = optimizer.init(params)
    args = (L,tissue_params,mesh_props,nc)

    iterator = tqdm(range(minimizer_params["maxiter"]))
    # A simple update loop.
    for i in iterator:
        grads = cost_grad_L(params, *args)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    print("iteration ", i, "E = ", get_E_L(params, *args))

    #
    # res = minimize(get_E,Y,jac=jac,args=(tissue_params,mesh_props,nc),method="Newton-CG")
    # Y = params

    return get_E_L(params, *args)#res.fun


tissue_params = {"kappa_A": 0.02,
                 "A0": 1.,
                 "P0": 9.67,
                 "kappa_P": 10**-2.7,
                 "T_ex": -0.03}

positive_indices = []
L_range = np.linspace(0.01,2,10)
gamma_val = 0.0257*1.5
# E_L = np.array([get_E_L_eq(L,positive_indices,pos_gamma_mult,0.0,_mesh_props,minimizer_params) for L in tqdm(L_range)])

fig, ax = plt.subplots()
positive_indices = []
E_L_1000 = np.array([get_E_L_eq(L,positive_indices,gamma_val,100,mesh_props,minimizer_params,tissue_params) for L in tqdm(L_range)])
ax.plot(L_range,E_L_1000)
positive_indices = [0]
E_L_1000 = np.array([get_E_L_eq(L,positive_indices,gamma_val,100,mesh_props,minimizer_params,tissue_params) for L in tqdm(L_range)])
ax.plot(L_range,E_L_1000)
positive_indices = [0,1,2,3]
E_L_1000 = np.array([get_E_L_eq(L,positive_indices,gamma_val,100,mesh_props,minimizer_params,tissue_params) for L in tqdm(L_range)])
ax.plot(L_range,E_L_1000)
positive_indices = [0,1,2,3,4,5]
E_L_1000 = np.array([get_E_L_eq(L,positive_indices,gamma_val,100,mesh_props,minimizer_params,tissue_params) for L in tqdm(L_range)])
ax.plot(L_range,E_L_1000)
# ax.set(ylim=(5,8))
# ax.set(yscale="log")
fig.show()


L_range = np.linspace(0.1,4,8)

A_vals1 = np.array([np.array([get_E_L_eq(L,np.nonzero(is_notch)[0],0.0257*1.5,0,mesh_props,minimizer_params,tissue_params) for is_notch in tqdm(is_notchs)]) for L in L_range])

mean_A_count = np.array([np.bincount(count,A_vals1[i])/np.bincount(count) for i in range(8)])



fig, ax = plt.subplots(figsize=(4,4))
for i in range(8):
    ax.plot(L_range,mean_A_count[:,i],color=plt.cm.plasma(i/8),label="%.2f"%(i/7))
ax.set(xlabel=r"$L$",ylabel="Energy")
ax.legend(title=r"$P_{Notch}$")
# ax.set(ylim=(0,10))

    # df_plot = pd.DataFrame({"A":A_vals1[i],"count":count})
    # sns.lineplot(ax=ax,data=df_plot,x="count",y="A",color=plt.cm.plasma(i/10))
fig.show()


A_vals2 = np.array([get_A_tot(np.nonzero(is_notch)[0],0.1,0.1,mesh_props,minimizer_params) for is_notch in is_notchs])
A_vals3 = np.array([get_A_tot(np.nonzero(is_notch)[0],0.1,1,mesh_props,minimizer_params) for is_notch in is_notchs])
A_vals4 = np.array([get_A_tot(np.nonzero(is_notch)[0],0.1,10,mesh_props,minimizer_params) for is_notch in is_notchs])
A_vals5 = np.array([get_A_tot(np.nonzero(is_notch)[0],0.1,100,mesh_props,minimizer_params) for is_notch in tqdm(is_notchs)])
A_vals6 = np.array([get_A_tot(np.nonzero(is_notch)[0],0.1,1000,mesh_props,minimizer_params) for is_notch in tqdm(is_notchs)])
A_vals7 = np.array([get_A_tot(np.nonzero(is_notch)[0],0.1,10000,mesh_props,minimizer_params) for is_notch in tqdm(is_notchs)])




"""
The conclusion here is that: 
- Under certain regimes, the 2D system becomes unstable shrinking to zero. 
- The volume constraint and basal coupling provides a means to buffer against the instability 
- When the constraint is set to extremely high; then little growth can happen at all. (but I think this could be hidden). 


To do: 
- over external p_notch, over the possible configurations, over Gamma
(a) Find the steady state 
(b) From the steady state, vary L over some range. 
(c) build the expected distribution over p_notch. 

NB: It feels as though this "lower bound" argument may be more generic. 
Try: 
1) Disable twisting of basal surfaces and repeat calculation. 


----

It is peculiar that the dz term is important when p_notch = 0 and the system is therefore isotropic. 
This is likely driven by higher tension at the perimeter. 

Yet this shouldn't be an issue; implying that the calculation for l_boundary/P is wrong. 


To do immediately:
(a) fix the border calculation 
(b) Improve calculation of dz / make it work properly. 



"""



##### TOY OF TOY MODEL OF CELL SIZE

def get_E_hexagon(l,tissue_params):
    A = 3*jnp.sqrt(3)/2 * l**2
    P = 6*l
    return tissue_params["kappa_A"]*(A-tissue_params["A0"])**2 + tissue_params["gamma"]*(P-tissue_params["P0"])**2

l_range = np.linspace(0,2)
E_L = np.array([get_E_hexagon(l,{"kappa_A":0.02,"A0":5.2,"gamma":1.3,"P0":+1}) for l in l_range])

fig, ax = plt.subplots()
ax.plot(l_range,E_L)
fig.show()



