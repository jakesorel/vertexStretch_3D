import sys
sys.dont_write_bytecode = True
import os
SCRIPT_DIR = "../../../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

from vertex_stretch_3d_periodic_rigid.simulation import Simulation,plot_3d,_plot_2d,get_vtx_by_cell
from vertex_stretch_3d_periodic_rigid.mesh import assemble_scalar
import numpy as np
import networkx as nx
from scipy import sparse
import pandas as pd
import pickle
import bz2
import os
from joblib import Parallel, delayed
import matplotlib
import seaborn as sns
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":

    p_notch_range = np.linspace(0.1,0.7,8)
    Ls = []
    L_saves = []
    sims = []
    for p_notch in p_notch_range:

        ##Improve meshing before running.

        # tissue_options = {"kappa_A": (0.02, 0.02),
        #                   "kappa_P": (0.003, 0.003),
        #                   "P0": (1,4.5),
        #                   "A0": (1.,1.),
        #                   "F_bend": (0.1,0.1),
        #                   "kappa_V": (0.3,0.3),
        #                   "V0": (1.0, 1.0),
        #                   "p_notch": p_notch,
        #                   "mu_l": 0.01,
        #                   "L_min": 0.1,
        #                   "T_external":-0.03,
        #                   "max_l_grad":10.}
        tissue_options = {"kappa_A": (0.02, 0.02),
                          "kappa_P": (0.003, 0.003),
                          "P0": (1,4.5),
                          "A0": (1.,1.),
                          "F_bend": (0.1,0.1),
                          "kappa_V": (0.3,0.3),
                          "V0": (1.0, 1.0),
                          "p_notch": p_notch,
                          "mu_l": 0.01,
                          "L_min": 0.1,
                          "T_external":-0.03,
                          "max_l_grad":10.}
        simulation_options = {"dt": 0.04,
                              "tfin": 1000,
                              "t_skip": 10}

        L = 10
        mesh_options = {"L": L, "A_init": (L / 9.05) ** 2, "V_init": 1., "init_noise": 4e-1, "eps": 0.002, "l_mult": 1.05,
                        "seed": 4}
        alpha = 0.05

        sim = Simulation(simulation_options, tissue_options, mesh_options)



        sim.simulate()
        sims.append(sim)
        # plt.scatter(*sim.t.mesh.mesh_props["r"][sim.t.tissue_params["is_notch"]].T)
        # plt.scatter(*sim.t.mesh.mesh_props["x"].T)
        # plt.show()
        L = sim.sim_out["L_save"][-1]
        Ls.append(L)
        L_saves.append(sim.sim_out["L_save"])
        print(L)

        colours = np.repeat(matplotlib.colors.rgb2hex(tuple(sns.color_palette("mako",as_cmap=True)(0.2))),sim.t.mesh.mesh_props["n_c"]).astype(object)
        colours[sim.t.tissue_params["is_notch"]] = matplotlib.colors.rgb2hex(tuple(sns.color_palette("mako",as_cmap=True)(0.8)))

        fig, ax = plt.subplots(1,2)
        cell_dict,vtx_position_dict,renormalised_X = get_vtx_by_cell(sim.t.mesh.mesh_props)
        _plot_2d(ax[0], renormalised_X, vtx_position_dict, colours,linewidth=3,edgecolor="white")
        ax[0].axis("off")
        ax[0].set(xlim=(-0.2,1.2),ylim=(-0.2,1.2))
        ax[1].plot(sim.sim_out["L_save"][:-1])
        # ax[1].set(ylim=(0,10))
        fig.show()


    """
    To do: 
    
    It would be sensible to initialise the tissue at a steady state size. 
    DONE
    
    Run the calculations for solving for dL / changes in volume given L again. I'm not 100% convinced. 
    This can be done quite simply within this package. 
    
    One could choose to opt for the plane definition rather than vertices; although I think this feels neater. 
    
    
    """

    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(p_notch_range,np.array(Ls)**2,color=sns.color_palette("mako", as_cmap=True)(0.6))
    ax.scatter(p_notch_range,np.array(Ls)**2,color=sns.color_palette("mako", as_cmap=True)(0.6))

    # ax.plot(p_notch_range,np.array(Ls))
    ax.set(xlabel="pNotch",ylabel="Tissue Area")
    fig.subplots_adjust(bottom=0.3,left=0.3,right=0.7,top=0.7)
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig("plots/tissue_area vs p_notch.pdf",dpi=300)

    for i, sim in enumerate(sims):
        if i > 0:
            mesh_props = sim.t.mesh.mesh_props
            is_notch = sim.t.tissue_params["is_notch"]
            tri = sim.t.mesh.mesh_props["tri"]
            t_is_notch = is_notch[tri].astype(int)
            n_v = assemble_scalar(np.ones_like(sim.t.mesh.mesh_props["tri"]),tri,int(np.max(tri)+1))
            n_notch_tri = t_is_notch.sum(axis=1)
            n_notch_neighbours = assemble_scalar(np.ones_like(tri)*np.expand_dims(n_notch_tri,1),tri,int(np.max(tri)+1))
            n_notch_neighbours = ((n_notch_neighbours - is_notch * n_v) / 2).astype(int)
            df_plot = pd.DataFrame({"n_notch_neighbours":n_notch_neighbours,
                                    "is_notch":is_notch,
                                    "A_apical":sim.t.mesh.mesh_props["A_apical"]*sim.t.effective_tissue_params["L"]**2,
                                    "R_basal":sim.t.mesh.mesh_props["R"][...,2]*sim.t.effective_tissue_params["L"]})

            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(data=df_plot,hue="is_notch",x="n_notch_neighbours",y="R_basal",whis=0, color="white", boxprops=dict(edgecolor="black", facecolor="none", linewidth=2), fliersize=0,legend=None)
            sns.stripplot(data=df_plot,hue="is_notch",x="n_notch_neighbours",y="R_basal",dodge=True,palette="mako")
            sns.violinplot(data=df_plot,hue="is_notch",x="n_notch_neighbours",y="R_basal",dodge=True,palette="mako",cut=0,legend=None,alpha=0.3)

            fig.subplots_adjust(bottom=0.3,left=0.3,right=0.8,top=0.8)
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_title("pNotch=%.3f"%p_notch_range[i])

            ax.legend(frameon=False)
            ax.set(xlabel="N Notch+ Neighbours",ylabel="Cell depth",xlim=(-0.5,5.5),ylim = (0.1,1.15))
            fig.savefig("plots/R_basal_p_notch=%.3f.pdf"%(p_notch_range[i]),dpi=300)

            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(data=df_plot,hue="is_notch",x="n_notch_neighbours",y="A_apical",whis=0, color="white", boxprops=dict(edgecolor="black", facecolor="none", linewidth=2), fliersize=0,legend=None)
            sns.stripplot(data=df_plot,hue="is_notch",x="n_notch_neighbours",y="A_apical",dodge=True,palette="mako")
            sns.violinplot(data=df_plot,hue="is_notch",x="n_notch_neighbours",y="A_apical",dodge=True,palette="mako",cut=0,legend=None,alpha=0.3)
            ax.set_title("pNotch=%.3f"%p_notch_range[i])
            fig.subplots_adjust(bottom=0.3,left=0.3,right=0.8,top=0.8)
            ax.spines[['right', 'top']].set_visible(False)
            ax.legend(frameon=False)
            ax.set(xlabel="N Notch+ Neighbours",ylabel="Cell Area",xlim=(-0.5,5.5),ylim = (0.8,6))
            fig.savefig("plots/A_apical_p_notch=%.3f.pdf"%(p_notch_range[i]),dpi=300)

            colours = np.repeat(matplotlib.colors.rgb2hex(sns.light_palette("seagreen", as_cmap=True)(0.3)), sim.t.mesh.mesh_props["n_c"]).astype(object)
            colours[sim.t.tissue_params["is_notch"]] = matplotlib.colors.rgb2hex(sns.light_palette("seagreen", as_cmap=True)(0.8))

            fig = plot_3d(sim.t.mesh.mesh_props, colours, aspectmode="data", zdirec=-1)
            fig.write_html("plots/p_notch=%.3f.html"%(p_notch_range[i]))
            fig.write_image("plots/p_notch=%.3f.pdf"%(p_notch_range[i]))

    ruggedness = []
    for i, sim in enumerate(sims):
        mesh_props = sim.t.mesh.mesh_props
        ruggedness.append(mesh_props["A_basal"].sum() / mesh_props["A_apical"].sum())
            # ruggedness.append(np.std(mesh_props["R"][...,2]*sim.t.effective_tissue_params["L"]))

    fig, ax = plt.subplots()
    ax.plot(p_notch_range,ruggedness)
    fig.show()


    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(p_notch_range,np.log(np.array(ruggedness)),color=sns.color_palette("mako", as_cmap=True)(0.6))
    ax.scatter(p_notch_range,np.log(np.array(ruggedness)),color=sns.color_palette("mako", as_cmap=True)(0.6))

    # ax.plot(p_notch_range,np.array(Ls))
    ax.set(xlabel="pNotch",ylabel="Ruggedness")
    fig.subplots_adjust(bottom=0.3,left=0.3,right=0.7,top=0.7)
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig("plots/Ruggedness vs p_notch.pdf",dpi=300)



    cluster_sizes = []
    for i, sim in enumerate(sims):
        mesh_props = sim.t.mesh.mesh_props
        # sim.t.mesh.get_adjacency()
        self = sim.t.mesh
        edges = np.row_stack(
            [np.column_stack([self.mesh_props["tri"][:, i], self.mesh_props["tri"][:, (i + 1) % 3]]) for i in
             range(3)])
        self.mesh_props["adj"] = sparse.coo_matrix((np.ones(len(edges), dtype=int), (edges[:, 0], edges[:, 1])),
                                                   shape=(int(self.mesh_props["n_c"]), int(self.mesh_props["n_c"])))

        adj = mesh_props["adj"]
        edges = np.row_stack(
            [np.column_stack([mesh_props["tri"][:, i], mesh_props["tri"][:, (i + 1) % 3]]) for i in range(3)])
        is_notch = sim.t.tissue_params["is_notch"]
        is_notch_int = sim.t.tissue_params["is_notch"].astype(int)
        is_notch_edges = is_notch_int[edges]

        adj_notch = sparse.coo_matrix((np.ones((is_notch_edges[:,0] == is_notch_edges[:,1]).sum(),dtype=int), (edges[(is_notch_edges[:,0] == is_notch_edges[:,1]), 0], edges[(is_notch_edges[:,0] == is_notch_edges[:,1]), 1])),
                                                   shape=(int(mesh_props["n_c"]), int(mesh_props["n_c"])))

        n_cc, cc_labs = sparse.csgraph.connected_components(adj_notch)


        cluster_size_negative = np.bincount(cc_labs[~is_notch])
        cluster_size_positive = np.bincount(cc_labs[is_notch])

        cluster_sizes.append([cluster_size_negative[cluster_size_negative!=0],cluster_size_positive[cluster_size_positive!=0]])


    cc_flat = []
    neg_pos = []
    p_n = []
    for i, cluster_sizs in enumerate(cluster_sizes):
        for j, negpos in enumerate(cluster_sizs):
            for val in negpos:
                cc_flat.append(val)
                neg_pos.append(j)
                p_n.append(i)
    df_cluster_size = pd.DataFrame({"Cluster Size":cc_flat,"Is Notch":neg_pos,"pNotch":p_n})
    df_cluster_size["Is Notch"][df_cluster_size["Is Notch"] == 0] = "Notch-"
    df_cluster_size["Is Notch"][df_cluster_size["Is Notch"] == 1] = "Notch+"

    fig, ax = plt.subplots(figsize=(6,4))
    sns.stripplot(ax = ax,data=df_cluster_size,x="pNotch",hue="Is Notch",y="Cluster Size",palette="mako")
    fig.subplots_adjust(bottom=0.3, left=0.3, right=0.8, top=0.8)
    ax.spines[['right', 'top']].set_visible(False)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    num_ticks = len(ax.get_xticks())

    ax.set_xticks(np.linspace(0,num_ticks-1,5))  # Setting ticks at positions 0, 1, 2, 3
    ax.set_xticklabels(["%.1f"%p for p in np.linspace(p_notch_range[0],p_notch_range[-1],5)])  # Setting tick labels

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),frameon=False)

    fig.savefig("plots/cluster size vs p_notch.pdf",dpi=300)

    max_cluster_sizes = np.array([np.array([np.max(cs) if len(cs)>0 else 0 for cs in css]) for css in cluster_sizes])

    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(p_notch_range, max_cluster_sizes[:,0],color=sns.color_palette("mako", as_cmap=True)(0.7))
    ax.plot(p_notch_range, max_cluster_sizes[:,1],color=sns.color_palette("mako", as_cmap=True)(0.2))
    ax.scatter(p_notch_range, max_cluster_sizes[:, 0], color=sns.color_palette("mako", as_cmap=True)(0.7), label="Notch-")
    ax.scatter(p_notch_range, max_cluster_sizes[:, 1], color=sns.color_palette("mako", as_cmap=True)(0.2), label="Notch+")

    fig.subplots_adjust(bottom=0.3, left=0.3, right=0.8, top=0.8)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(frameon=False)
    ax.set(xlabel="pNotch", ylabel="Giant Cluster Size")

    fig.savefig("plots/giant_cluster_size.pdf",dpi=300)