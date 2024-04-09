from jax import numpy as jnp
from jax import jit, hessian, jacrev
from scipy.optimize import minimize
from tqdm import tqdm
import seaborn as sns
import pandas as pd



@jit
def periodic_displacement(x1, x2, L):
    return jnp.mod(x1 - x2 + L / 2, L) - L / 2

@jit
def get_E(z,tz_0,x,y,neigh,neighp1,neighm1,k2s,beta,L):
    X = jnp.column_stack([x,y,z])

    Xp1 = X[neighp1]
    Xm1 = X[neighm1]
    _X = jnp.expand_dims(X,axis=1)

    norm = jnp.cross(periodic_displacement(Xm1,_X,L),periodic_displacement(Xp1,_X,L),axis=2)
    norm_neigh = norm[neigh,k2s]
    norm_p1 = jnp.roll(norm_neigh, 1, axis=1)
    norm_m1 = jnp.roll(norm_neigh, -1, axis=1)
    norm_norm = jnp.linalg.norm(norm,axis=-1)
    norm_norm_p1 = jnp.roll(norm_norm,1,axis=1)
    norm_norm_m1 = jnp.roll(norm_norm,-1,axis=1)

    dot_p1 = (norm*norm_p1).sum(axis=-1)/norm_norm/norm_norm_p1
    dot_m1 = (norm*norm_m1).sum(axis=-1)/norm_norm/norm_norm_m1

    return ((jnp.expand_dims(z,1) - tz_0)**2).sum() + beta*(((dot_p1-1)**2).sum()+((dot_m1-1)**2).sum())

jac = jit(jacrev(get_E))
hess = jit(hessian(get_E))

T_cortical, alpha, A0, p_notch, file_name, seed = (
0.1909090909090909, 0.4545454545454546, 7.045454545454546, 0.3, 'results', 3)

p_notch_range = np.linspace(0,0.7,6)
n_iter = 2
z_mean = np.zeros((len(p_notch_range),n_iter))
for k, p_notch in tqdm(enumerate(p_notch_range)):
    for q in range(n_iter):

        mesh_options = {"L": 18.4, "A_init": 2., "V_init": 1., "init_noise": 1e-1, "eps": 0.002, "l_mult": 1.05,
                        "seed": q + 2024}
        tissue_options = {"kappa_A": (0.02, 0.02),
                          "A0": (A0, A0),
                          "T_lateral": (0., 0. * alpha),
                          "T_cortical": (T_cortical, T_cortical * alpha),
                          "T_basal": (0., 0.),
                          "F_bend": (1., 1.),
                          "T_external": 0.,
                          "kappa_V": (1.0, 1.0),
                          "V0": (1.0, 1.0),
                          "p_notch": p_notch,
                          "mu_L": 10.,  ##beware i changed form 0.25
                          "L_min": 2.,
                          "max_L_grad": 100}
        simulation_options = {"dt": 0.01,
                              "tfin": 40,
                              "t_skip": 10}
        sim = Simulation(simulation_options, tissue_options, mesh_options)

        mesh_props = sim.t.mesh.mesh_props
        tri = mesh_props["tri"]
        is_notch = sim.t.tissue_params["is_notch"]

        z0_P = 1
        z0_N = 0.5

        z_init = 0.5*np.ones(len(tri))

        tis_notch = is_notch[tri]
        t_z0 = np.zeros_like(tis_notch,dtype=float)
        t_z0[tis_notch == 0] = z0_N
        t_z0[tis_notch == 1] = z0_P

        cell_to_vertex = np.array([(tri==i).any(axis=1) for i in range(np.max(tri)+1)])
        cell_to_vertex_list = [np.nonzero(msk)[0] for msk in cell_to_vertex]


        x = mesh_props["x"][:,0]
        y = mesh_props["x"][:,1]

        neighp1 = jnp.roll(mesh_props["neigh"], -1, axis=1)
        neighm1 = jnp.roll(mesh_props["neigh"], 1, axis=1)

        neigh,k2s = mesh_props["neigh"],mesh_props["k2s"]

        L = mesh_props["L"]



        res = minimize(get_E,z_init,args=(t_z0,x,y,neigh,neighp1,neighm1,k2s,0.2,L),jac=jac,hess=hess,method="Newton-CG",options={"xtol":1e-7})
        z_mean[k,q] = res.x.mean()


x = mesh_props["x"]
fig, ax = plt.subplots()
ax.scatter(*x.T,c=res.x)
ax.scatter(*mesh_props["r"].T,c=plt.cm.plasma(is_notch.astype(float)),alpha=0.5,s=20)
fig.show()



df = pd.DataFrame(z_mean.T)
df.columns = p_notch_range

fig, ax = plt.subplots()
sns.lineplot(data=df.melt(),x="variable",y="value",ax=ax)
ax.set(xlabel="pNotch",ylabel="Average Basal Vertex Length")
ax.plot(p_notch_range,0.5+0.5*p_notch_range)
fig.show()


"""
Corrolary: 
- While very stiff, in theory, the tiling can be accomodated
- This does indeed lead to propagation of tilt
- Note: it may be possible, but is it optimal? 
Is there a way to cast this with growth to survey the effects in a minimal setup 


"""
