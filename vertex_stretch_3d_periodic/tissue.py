import sys
sys.dont_write_bytecode = True
import os
SCRIPT_DIR = "../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import jax.numpy as jnp
from jax import jacrev
from jax import jit
from functools import partial
import jax
from vertex_stretch_3d_periodic.mesh import Mesh,get_geometry
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)

class Tissue:
    def __init__(self,tissue_options=None,true_mesh_options=None):
        assert true_mesh_options is not None
        assert tissue_options is not None
        self.true_mesh_options = true_mesh_options

        ##Make an effective mesh in a box of size 1x1
        mesh_options = true_mesh_options.copy()
        mesh_options["L"] = 1
        mesh_options["A_init"] = true_mesh_options["A_init"]/true_mesh_options["L"]**2
        mesh_options["V_init"] = true_mesh_options["V_init"]/true_mesh_options["L"]**3
        mesh_options["init_noise"] = true_mesh_options["init_noise"]/true_mesh_options["L"]

        self.mesh = Mesh(mesh_options)
        self.tissue_options = tissue_options
        self.tissue_params = {}
        self.effective_tissue_params = {}
        self.jac = None

        self.initialize()

    def initialize(self):
        self.assign_notch_positive_cells()
        self.assign_tissue_parameters()
        self.initialize_energies()
        # self.initialize_L()
        self.update_effective_tissue_params()
        # print(self.effective_tissue_params["L"])

    def assign_notch_positive_cells(self):
        self.tissue_params["n_notch"] = int(np.round(self.tissue_options["p_notch"]*self.mesh.mesh_props["n_c"]))
        self.tissue_params["is_notch"] = np.zeros((int(self.mesh.mesh_props["n_c"])),dtype=np.bool_)
        self.tissue_params["is_notch"][:self.tissue_params["n_notch"]] = True
        np.random.seed(self.mesh.mesh_options["seed"])
        np.random.shuffle(self.tissue_params["is_notch"])

    def initialize_energies(self):
        E = energy(self.mesh.mesh_props["X"], self.mesh.mesh_props, int(self.mesh.mesh_props["n_c"]), self.effective_tissue_params)
        self.jac = jit(jacrev(energy),static_argnums=(2,))

    def assign_tissue_parameters(self):
        self.tissue_params["A0"] = np.ones((int(self.mesh.mesh_props["n_c"])))*self.tissue_options["A0"][0]
        self.tissue_params["A0"][self.tissue_params["is_notch"]] = self.tissue_options["A0"][1]

        self.tissue_params["V0"] = np.ones((int(self.mesh.mesh_props["n_c"])))*self.tissue_options["V0"][0]
        self.tissue_params["V0"][self.tissue_params["is_notch"]] = self.tissue_options["V0"][1]

        self.tissue_params["T_lateral"] = np.ones((int(self.mesh.mesh_props["n_c"]))) * self.tissue_options["T_lateral"][0]
        self.tissue_params["T_lateral"][self.tissue_params["is_notch"]] = self.tissue_options["T_lateral"][1]

        self.tissue_params["T_cortical"] = np.ones((int(self.mesh.mesh_props["n_c"]))) * self.tissue_options["T_cortical"][0]
        self.tissue_params["T_cortical"][self.tissue_params["is_notch"]] = self.tissue_options["T_cortical"][1]

        self.tissue_params["T_basal"] = np.ones((int(self.mesh.mesh_props["n_c"]))) * self.tissue_options["T_basal"][0]
        self.tissue_params["T_basal"][self.tissue_params["is_notch"]] = self.tissue_options["T_basal"][1]

        self.tissue_params["F_bend"] = np.ones((int(self.mesh.mesh_props["n_c"]))) * self.tissue_options["F_bend"][0]
        self.tissue_params["F_bend"][self.tissue_params["is_notch"]] = self.tissue_options["F_bend"][1]

        self.tissue_params["kappa_A"] = np.ones((int(self.mesh.mesh_props["n_c"]))) * self.tissue_options["kappa_A"][0]
        self.tissue_params["kappa_A"][self.tissue_params["is_notch"]] = self.tissue_options["kappa_A"][1]

        self.tissue_params["kappa_V"] = np.ones((int(self.mesh.mesh_props["n_c"]))) * self.tissue_options["kappa_V"][0]
        self.tissue_params["kappa_V"][self.tissue_params["is_notch"]] = self.tissue_options["kappa_V"][1]

        self.tissue_params["mu_L"] = self.tissue_options["mu_L"]
        self.tissue_params["T_external"] = self.tissue_options["T_external"]
        self.tissue_params["L"] = self.true_mesh_options["L"]
        self.effective_tissue_params = self.tissue_params.copy()

    def update_effective_tissue_params(self):
        ##prefix parameters are rescaled by 1/L^2 to account for the effective damping coefficient of 1/L^2
        self.effective_tissue_params["kappa_V"] = self.tissue_params["kappa_V"]*self.effective_tissue_params["L"]**4
        self.effective_tissue_params["kappa_A"] = self.tissue_params["kappa_A"]*self.effective_tissue_params["L"]**2
        self.effective_tissue_params["T_cortical"] = self.tissue_params["T_cortical"]/self.effective_tissue_params["L"]

        ##Note that T_lateral and T_basal are the same upon rescaling and damping coefficient factoring L^2/L^2 = 1

        ##These are not scaled by damping coefficients
        self.effective_tissue_params["V0"] = self.tissue_params["V0"]/self.effective_tissue_params["L"]**3
        self.effective_tissue_params["A0"] = self.tissue_params["A0"]/self.effective_tissue_params["L"]**2

        ##F_bend considers angles and is thus scale free.

    # def initialize_L(self):
    #     self.effective_tissue_params["L"] = fsolve(dE_dL, 17.0, args=(self.mesh.mesh_props, self.tissue_params))[0]
    #     print(self.effective_tissue_params["L"])

    def update_L(self,dt):
        self.effective_tissue_params["L"] += - dt*self.effective_tissue_params["mu_L"]*dE_dL(self.effective_tissue_params["L"],self.mesh.mesh_props,self.tissue_params)
        if self.effective_tissue_params["L"] < self.tissue_options["L_min"]:
            self.effective_tissue_params["L"] = self.tissue_options["L_min"]
        self.update_effective_tissue_params()

    def update_x(self,X,dt):
        F = -self.jac(X,self.mesh.mesh_props,int(self.mesh.mesh_props["n_c"]),self.effective_tissue_params)
        X += dt*F
        X = jnp.mod(X,1.)
        return X

    def update(self,X,dt):
        """Some care may need to be given regarding the ordering here"""
        self.mesh.update_mesh_props(X)
        self.update_L(dt)
        X = self.update_x(X,dt)
        X = self.mesh.perform_T1s(X)
        return X, self.effective_tissue_params["L"]


@partial(jit, static_argnums=(2,))
def energy(X,mesh_props,nc,tissue_params):
    mesh_props = get_geometry(X,mesh_props,nc)

    E = tissue_params["kappa_V"]*(mesh_props["V"]-tissue_params["V0"])**2 \
        + tissue_params["kappa_A"]*(mesh_props["A_apical"]-tissue_params["A0"])**2 \
        + tissue_params["T_lateral"] * mesh_props["A_lateral"] \
        + tissue_params["T_basal"] * mesh_props["A_basal"] \
        + tissue_params["F_bend"] * (mesh_props["phi"] - jnp.pi*2)**2 \
        + tissue_params["T_external"]*mesh_props["A_apical"] \
        + tissue_params["T_cortical"]*mesh_props["P_apical"]

    return E.sum()

@jit
def dE_dL(L,mesh_props,tissue_params):
    A0 = tissue_params["A0"]
    ka = tissue_params["kappa_A"]
    Tb = tissue_params["T_basal"]
    Tl = tissue_params["T_lateral"]
    Tex = tissue_params["T_external"]
    kv = tissue_params["kappa_V"]
    V0 = tissue_params["V0"]
    Tcortical = tissue_params["T_cortical"]

    a = mesh_props["A_apical"]
    al = mesh_props["A_lateral"]
    ab = mesh_props["A_basal"]
    v = mesh_props["V"]
    p = mesh_props["P_apical"]
    dE_dLi =  p*Tcortical + 2*L*(2*a**2*ka*L**2 + ab*Tb + a*(-2*A0*ka + Tex) + al*Tl + 3*kv*L*v*(L**3*v - V0))
    return dE_dLi.sum()


# if __name__ == "__main__":
    # mesh_options = {"L": 18.4, "A_init": 2., "V_init": 1., "init_noise": 1e-1,"eps":0.002,"l_mult":1.05}
    # tissue_options = {"kappa_A":(0.1,0.1),
    #                   "A0":(2.,3.),
    #                   "T_lateral":(0.05,0.05),
    #                   "T_basal":(0.,0.),
    #                   "F_bend":(0.1,0.1),
    #                   "T_external":-0.05,
    #                   "kappa_V":(1.0,1.0),
    #                   "V0":(1.0,1.0),
    #                   "p_notch":0.2,
    #                   "mu_L":3.,
    #                   "L_min":6.}
    # t = Tissue(tissue_options,mesh_options)
    #
    # X = t.mesh.mesh_props["X"].copy()
    # mesh_props = t.mesh.mesh_props
    #
    # plt.scatter(*mesh_props["x"].T)
    # plt.scatter(*mesh_props["r"].T)
    # plt.show()
    #
    # dt = 0.01
    # tfin = dt*300
    # t_span = np.arange(0,tfin,dt)
    #
    # X_save = np.zeros((len(t_span),len(X),3))
    # L_save = np.zeros((len(t_span)))
    # for i, tm in enumerate(t_span):
    #     X,L = t.update(X,dt)
    #     X_save[i] = X.copy()
    #     L_save[i] = L
    #
    # fig, ax = plt.subplots()
    # ax.scatter(*X_save[0,:,:2].T)
    # ax.scatter(*X_save[-1,:,:2].T)
    # fig.show()
    #
    # def get_L_eq(p_notch):
    #     mesh_options = {"L": 18.4, "A_init": 2., "V_init": 1., "init_noise": 1e-1, "eps": 0.002, "l_mult": 1.05}
    #     tissue_options = {"kappa_A": (0.1, 0.03),
    #                       "A0": (2., 2.),
    #                       "T_lateral": (0.05, 0.05),
    #                       "T_basal": (0.0, 0.0),
    #                       "F_bend": (1.0, 1.0),
    #                       "T_external": -0.2,
    #                       "kappa_V": (1.0, 1.0),
    #                       "V0": (1.0, 1.0),
    #                       "p_notch": p_notch,
    #                       "mu_L": 4.,
    #                       "L_min": 6.}
    #     t = Tissue(tissue_options, mesh_options)
    #
    #     X = t.mesh.mesh_props["X"].copy()
    #
    #
    #
    #     dt = 0.01
    #     tfin = dt * 300
    #     t_span = np.arange(0, tfin, dt)
    #
    #     X_save = np.zeros((len(t_span), len(X), 3))
    #     L_save = np.zeros((len(t_span)))
    #     for i, tm in enumerate(t_span):
    #         X, L = t.update(X, dt)
    #         X_save[i] = X.copy()
    #         L_save[i] = L
    #     mesh_props = t.mesh.mesh_props
    #
    #     plt.scatter(*mesh_props["x"].T,c=mesh_props["X"][...,2])
    #     # plt.scatter(*mesh_props["r"].T)
    #     plt.show()
    #     return L_save[-1],X_save[-1]
    #
    # Ls,Xs = [],[]
    # p_notch_range = np.linspace(0,1,5)
    # for i, p_notch in enumerate(p_notch_range):
    #     L,X = get_L_eq(p_notch)
    #     Ls.append(L)
    #     print(L)
    #     Xs.append(X)
    #
    #
