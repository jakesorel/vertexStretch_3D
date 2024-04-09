import numpy as np
from jax import jit, jacrev,hessian
import matplotlib.pyplot as plt

"""
Proposition: 
- With fixed cell centres, allow radii of cells to vary. 
- Build a power diagram, and calculate areas, perimeters etc
- Consider different degrees of basal coupling 
- As a proxy for a power diagram, could calculate the effective radiius of cells 
    d_1 = (d^2 - R_2^2 + R_1^2)/(2d)
    R_eff = mean(d_1)

"""



import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
import jax
from scipy import sparse
import triangle as tr

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)


class Mesh:
    """
    Let x be the 2D vertex position
    let z be the basal z distance of that vertex
    Let X be the 3D vector (x,z)
    Let r be the centre of mass of each cell, in 2D
    Let r_z be the z value of the centre of mass
    let R be the centre of mass of each cell in 3D: R = (r, r_z).
    """

    def __init__(self, mesh_options=None):
        assert mesh_options is not None
        self.mesh_options = mesh_options
        self.mesh_props = {}
        self.initialize()

    def initialize(self):
        self.generate_hexagonal()
        self.voronoi_triangulate()

    def generate_hexagonal(self):
        """
        Generate a hexagonal grid such that each cell has on average an area A_init
        (supposing cells follow a voronoi about cell centres r)
        """
        r = hexagonal_lattice(int(self.mesh_options["L"]), int(np.ceil(self.mesh_options["L"])),
                              noise=self.mesh_options["init_noise"], A=self.mesh_options["A_init"],seed=self.mesh_options["seed"])
        r += 1e-3
        r = r[np.argsort(r.max(axis=1))[:int(self.mesh_options["L"] ** 2 / self.mesh_options["A_init"])]].astype(
            np.float32)
        r += self.mesh_options["L"]/4
        r = np.mod(r,self.mesh_options["L"])

        self.mesh_props["L"] = self.mesh_options["L"]
        self.mesh_props["r"] = r

    def voronoi_triangulate(self):
        """
        Initialize vertices as the voronoi triangulation of cell centres
        """
        y, dictionary = generate_triangulation_mask(self.mesh_props["r"], self.mesh_props["L"], self.mesh_props["L"])
        t = tr.triangulate({"vertices": y})
        tri = t["triangles"]
        self.mesh_props["n_c"] = self.mesh_props["r"].shape[0]
        tri = tri[(tri != -1).all(axis=1)]
        one_in = (tri < self.mesh_props["n_c"]).any(axis=1)
        new_tri = tri[one_in]
        n_tri = dictionary[new_tri]
        n_tri = remove_repeats(n_tri, self.mesh_props["n_c"])
        self.mesh_props["tri"] = n_tri
        self.mesh_props["n_v"] = self.mesh_props["tri"].shape[0]
        self.mesh_props["neigh"] = get_neighbours(self.mesh_props["tri"])
        self.mesh_props["t_r"] = self.mesh_props["r"][self.mesh_props["tri"]]
        self.mesh_props["x"] = jnp.array(circumcenter(self.mesh_props["t_r"], self.mesh_props["L"]))
        self.mesh_props["k2s"] = get_k2(self.mesh_props["tri"], self.mesh_props["neigh"])



def generate_triangulation_mask(x, L, max_d):
    ys = jnp.zeros((0, 2), dtype=jnp.float32)
    dictionary = jnp.zeros((0), dtype=jnp.int32)
    for i in [0, -1, 1]:
        for j in [0, -1, 1]:
            y = (x + np.array((i, j)) * L).astype(jnp.float32)
            if j == 0:
                if i == 0:
                    mask = jnp.ones_like(x[:, 0], dtype=jnp.bool_)
                else:
                    val = L * (1 - i) / 2
                    mask = jnp.abs(x[:, 0] - val) < max_d
            elif i == 0:
                val = L * (1 - j) / 2
                mask = jnp.abs(x[:, 1] - val) < max_d
            else:
                val_x = L * (1 - i) / 2
                val_y = L * (1 - j) / 2
                mask = jnp.sqrt((x[:, 0] - val_x) ** 2 + (x[:, 1] - val_y) ** 2) < max_d
            ys = jnp.row_stack((ys, y[mask]))
            dictionary = jnp.concatenate((dictionary, jnp.nonzero(mask)[0].astype(jnp.int32)))
    return ys, dictionary


def order_tris(tri):
    """
    For each triangle (i.e. row in **tri**), order cell ids in ascending order
    :param tri: Triangulation (n_v x 3) np.int32 array
    :return: the ordered triangulation
    """
    nv = tri.shape[0]
    for i in range(nv):
        Min = np.argmin(tri[i])
        tri[i] = tri[i, Min], tri[i, np.mod(Min + 1, 3)], tri[i, np.mod(Min + 2, 3)]
    return tri


def remove_repeats(tri, n_c):
    """
    For a given triangulation (nv x 3), remove repeated entries (i.e. rows)
    The triangulation is first re-ordered, such that the first cell id referenced is the smallest. Achieved via
    the function order_tris. (This preserves the internal order -- i.e. CCW)
    Then remove repeated rows via lexsort.
    NB: order of vertices changes via the conventions of lexsort
    Inspired by...
    https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array
    :param tri: (nv x 3) matrix, the triangulation
    :return: triangulation minus the repeated entries (nv* x 3) (where nv* is the new # vertices).
    """
    tri = order_tris(np.mod(tri, n_c))
    sorted_tri = tri[np.lexsort(tri.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_tri, axis=0), 1))
    return sorted_tri[row_mask]


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
                neighb, l = np.nonzero((tri_compare[:, :, 0] == tri_i[k, 0]) * (tri_compare[:, :, 1] == tri_i[k, 1]))
                neighb, l = neighb[0], l[0]
                neigh[j, k] = neighb
                neigh[neighb, np.mod(2 - l, 3)] = j
    return neigh.astype(np.int32)


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

#
# @partial(jit, static_argnums=(1,))
# def get_R(mesh_props, nc):
#     mesh_props = get_2d_COM(mesh_props, nc)
#     mesh_props = get_rz(mesh_props, nc)
#     mesh_props["R"] = jnp.column_stack((mesh_props["r"], mesh_props["r_z"]))
#     mesh_props["tR"] = mesh_props["R"][mesh_props["tri"]]
#     return mesh_props
#
#
# @partial(jit, static_argnums=(1,))
# def get_2d_COM(mesh_props, nc):
#     t_r = (mesh_props["X_R"] + mesh_props["Xp1_R"]) * jnp.expand_dims(
#         mesh_props["tri_A_apical"] / (3 * mesh_props["tA_apical"] + 1e-17), 2)
#     mesh_props["n_v"] = assemble_scalar(jnp.ones_like(mesh_props["X"]), mesh_props["tri"], nc)
#     r_old = mesh_props["R"][..., :2]
#     r_new = -2 * r_old / jnp.expand_dims(mesh_props["n_v"], 1) + jnp.column_stack(
#         [assemble_scalar(t_r[..., 0], mesh_props["tri"], nc),
#          assemble_scalar(t_r[..., 1], mesh_props["tri"], nc)])
#
#     mesh_props["r"] = jnp.mod(r_new, mesh_props["L"])
#     return mesh_props


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


def circumcenter(C, L):
    """
    Find the circumcentre (i.e. vertex position) of each triangle in the triangulation.

    :param C: Cell centroids for each triangle in triangulation (n_c x 3 x 2) np.float32 array
    :param L: Domain size (np.float32)
    :return: Circumcentres/vertex-positions (n_v x 2) np.float32 array
    """
    ri, rj, rk = C.transpose(1, 2, 0)
    r_mean = (ri + rj + rk) / 3
    disp = r_mean - L / 2
    ri, rj, rk = np.mod(ri - disp, L), np.mod(rj - disp, L), np.mod(rk - disp, L)
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
    vs = np.mod(vs + disp.T, L).astype(np.float32)
    return vs



@partial(jit, static_argnums=(2,))
def assemble_scalar(tval, tri, nc):
    val = jnp.bincount(tri.ravel() + 1, weights=tval.ravel(), length=nc + 1)
    val = val[1:]
    # val = jnp.zeros(nc)
    # for i in range(3):
    #     val = val.at[tri[..., i]].add(tval[..., i])
    return val

@jit
def periodic_displacement(x1, x2, L):
    return jnp.mod(x1 - x2 + L / 2, L) - L / 2


mesh_options = {"L": 18.4, "A_init": 2., "V_init": 1., "init_noise": 1e-3, "eps": 0.002, "l_mult": 1.05,
                "seed": 1 + 2024}

L = mesh_options["L"]
msh = Mesh(mesh_options)
tri = msh.mesh_props["tri"]
print((np.bincount(tri.ravel())!=6).sum())
x = msh.mesh_props["x"]
neigh = msh.mesh_props["neigh"]
k2s = msh.mesh_props["k2s"]
t_r = msh.mesh_props["r"][tri]
x_tx = periodic_displacement(np.expand_dims(x,1)*np.ones((len(x),3,2)),t_r,L)
n_c = np.max(tri)+1
nv = len(tri)

M_x = np.zeros((nv,n_c,3))
M_y = np.zeros((nv,n_c,3))

edges = np.column_stack([[tri[:,k],tri[:,(k+1)%3]] for k in range(2)]).T
M_x = sparse.coo_matrix((np.concatenate([x[:,0],-x[:,0]]),(np.arange(len(edges)),edges[:,0])),shape=(len(edges),n_c))
M_y = sparse.coo_matrix((np.concatenate([x[:,1],-x[:,1]]),(np.arange(len(edges)),edges[:,0])),shape=(len(edges),n_c))

M = np.column_stack((M_x.toarray(),M_y.toarray()))

gamma = np.random.random(n_c)
dgammas = np.concatenate([gamma[tri[:,k+1]]-gamma[tri[:,k]] for k in range(2)])

alpha_beta = np.linalg.pinv(M)@dgammas
alpha = alpha_beta[:169]
beta = alpha_beta[169:]









theta = np.arctan2(x_tx[...,1],x_tx[...,0])
tri_labs = np.bincount(np.where((theta-np.pi/3)**2 < 0.5**2)[0],minlength=np.max(tri)+1)

z_vals = np.random.random(169)
reindexed_vertices = np.cumsum(tri_labs)-1

z_vals_filled = np.zeros((169,169*2))

for i in range(169):
    vs_in_cell = np.nonzero(tri == i)[0]
    chosen_vs_in_cell = vs_in_cell[tri_labs[vs_in_cell].astype(bool)]
    reindexed_vs_in_cell = reindexed_vertices[chosen_vs_in_cell]
    z = z_vals[reindexed_vs_in_cell]
    _x = x[chosen_vs_in_cell]
    X = np.column_stack([_x,z])
    dX = X[1:]-X[0]
    norm = np.cross(dX[0],dX[1])
    norm /= np.linalg.norm(norm)
    p = np.dot(X[0],norm)
    ##ax + by + cz = d
    ##--> z = (d - ax - by)/c
    z_eqn = np.array([p,-norm[0],-norm[1]])/norm[2]

    _X = np.column_stack([np.ones(vs_in_cell.size),x[vs_in_cell]])
    z_cell = _X@z_eqn
    z_vals_filled[i,vs_in_cell] = z_cell



std_z_vals = np.array([z_vals_filled[:,i][z_vals_filled[:,i]!=0].std() for i in range(169*2)])

plt.scatter(*x.T,c=(std_z_vals>0.0005))
# plt.scatter(*x.T,c=tri_labs)

plt.show()




### Toy model attempt
sim.t.mesh

