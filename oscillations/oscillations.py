
import numpy as np
import jax.numpy as jnp
from jax import jit,vmap,jacrev,hessian,grad
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



@jit
def get_E_hexagon(l,A0,tissue_params):
    A = 3*jnp.sqrt(3)/2 * l**2
    P = 6*jnp.abs(l)
    return tissue_params["kappa_A"]*(A-A0)**2 + tissue_params["gamma"]*P

@jit
def get_F(l,A0,tissue_params):
    return - grad(get_E_hexagon)(l,A0,tissue_params)

dt = 0.01
tfin = 20
t_span = np.arange(0,tfin,dt)

amp = 3
period = 20
A0_t = 5.2 + amp*np.sin(t_span*2*np.pi/period)

def get_l_t(l0,t_span,A0_t,tissue_params):
    l_save = np.zeros_like(t_span)
    l = l0
    for i,t in enumerate(tqdm(t_span)):
        F = get_F(l,A0_t[i],tissue_params)
        l += F*dt
        l_save[i] = l
    return l_save


fig, ax = plt.subplots()
gamma_range = np.linspace(0.013,0.05,10)
for i, gamma in enumerate(gamma_range):
    ax.plot(t_span,get_l_t(1.,t_span,A0_t,{"kappa_A":0.02,"gamma":gamma}),color=plt.cm.plasma(i/len(gamma_range)))

fig.show()

def get_l_t_max(l0,t_span,A0_t,tissue_params):
    l_save = np.zeros_like(t_span)
    l = l0
    dt = t_span[1]-t_span[0]
    for i,t in enumerate(t_span):
        F = get_F(l,A0_t[i],tissue_params)
        l += F*dt
        l_save[i] = l
    return l_save[-int(t_span.size/2):].max()

N = 8
A0_range = np.linspace(3,8,N)
period_range = np.logspace(-1,1,N)
amp_range = np.linspace(0,5,N)
gamma_range = np.linspace(0.005,0.13,N)

amp = 3
l_t_max = np.zeros((N,N))
# for i, A0 in enumerate(tqdm(A0_range)):
for j, gamma in enumerate(tqdm(gamma_range)):
    for k, period in enumerate(period_range):
        A0_t = 5.2 + amp * np.sin(t_span * 2 * np.pi / period)
        l_t_max[j,k] = get_l_t_max(1.,t_span,A0_t,{"kappa_A":0.02,"gamma":gamma})


fig, ax = plt.subplots()
ax.imshow(l_t_max[:,:])
fig.show()

fig, ax = plt.subplots()
for k, period in enumerate(period_range):
    ax.plot(l_t_max[:,k],color=plt.cm.plasma(k/8))
fig.show()


@jit
def get_E_A(A,A0):
    return A*(1.0*(A>A0) + (2*A/A0 - 1)*(A<=A0))

@jit
def get_E_hexagon(l,A0,P0,gamma,Te):
    A = 3*jnp.sqrt(3)/2 * l**2
    P = 6*jnp.abs(l)
    return 0.02*get_E_A(A,A0) + gamma*(P-P0)**2 + Te*A
    # return 0.02*(A-A0)**2 + gamma*(P-P0)**2 + Te*A + adh*P/A


@jit
def get_F(l,A0,P0,gamma,Te):
    return - grad(get_E_hexagon)(l,A0,P0,gamma,Te)

alpha = 0
beta = 0.1

l_range = np.linspace(0,4*np.sqrt(50/(3*jnp.sqrt(3)/2)),200)
l_vals = np.zeros((40,2))
for i, P0 in enumerate(np.linspace(10,50,40)):
    args = (50,P0,10**-2.7,-0.025)
    e = get_E_hexagon(l_range,*args)
    l_vals[i,0] = l_range[:50][np.argmin(e[:50])]
    l_vals[i,1] = l_range[50:][np.argmin(e[50:])]

    e0 = e.min()#get_E_hexagon(np.sqrt(50/(3*jnp.sqrt(3)/2)), *args)
    plt.plot(l_range,np.log((e-e0)/(e.max()-e0)+0.1),color=plt.cm.plasma(i/40))
# plt.plot(l_range,get_E_hexagon(l_range,50,30,0.03,-0.03))
plt.show()

l_vals[l_vals[:,1]==l_vals[0,1],1] = np.nan
plt.plot(np.nanmax(l_vals,axis=1))
plt.show()


fprime = jit(jacrev(get_E_hexagon))

def get_fprime(l,A0,P0,gamma,Te):
    return np.array(fprime(l,A0,P0,gamma,Te).reshape(-1,1))


from scipy.optimize import fsolve



l_vals = np.zeros((10,10))
l0_range = np.linspace(2,20,10)
for i, P0 in enumerate(np.linspace(1,40,10)):
    for j, l0 in enumerate(l0_range):
        args = (50,P0,10**-2.5,-0.03)
        l_vals[i,j] = fsolve(get_E_hexagon, l0, args=args)#, fprime=get_fprime)[0]

l0 = np.sqrt(50/(3*jnp.sqrt(3)/2))

fig, ax = plt.subplots()
for i in range(10):
    ax.plot(np.linspace(1, 40, 10), l_vals[:,i])
fig.show()




###Fake population


@jit
def get_E_hexagon(l,A0,P0_N,P0_P,gamma,Te,p_notch):
    ##This would work under the assumption that all cells are the same shape.
    A = 3*jnp.sqrt(3)/2 * l**2
    P = 6*jnp.abs(l)
    return 0.02*get_E_A(A,A0) + p_notch*gamma*(P-P0_P)**2 + (1-p_notch)*gamma*(P-P0_N)**2 + Te*A
    # return 0.02*(A-A0)**2 + gamma*(P-P0)**2 + Te*A + adh*P/A

##gamma = ~0.0257

args = [1.,0.,9.67,10**-2.7,-0.03,0.]
l_range = np.linspace(0,4*np.sqrt(args[0]/(3*jnp.sqrt(3)/2)),800)
p_notch_range = np.linspace(0.1,1,20)
l_vals = np.zeros((p_notch_range.shape[0],2))

for i, p_notch in enumerate(p_notch_range):
    args[-1] = p_notch
    e = get_E_hexagon(l_range,*args)
    l_vals[i,0] = l_range[:200][np.argmin(e[:200])]
    l_vals[i,1] = l_range[200:][np.argmin(e[200:])]

    e0 = e.min()#get_E_hexagon(np.sqrt(50/(3*jnp.sqrt(3)/2)), *args)
    plt.plot(l_range,np.log((e-e0)/(e.max()-e0)+0.1),color=plt.cm.plasma(i/10))
# plt.plot(l_range,get_E_hexagon(l_range,50,30,0.03,-0.03))
plt.show()



l_vals[l_vals[:,1]==l_vals[0,1],1] = np.nan
plt.plot(np.nanmax(l_vals,axis=1))
plt.show()


"""
OK so under a certain regime, this energy functional yields a bistable response. 
The core benefit is that there's a penalty to small cell areas, preventing cells from collapsing. 
I think that the basal coupling we see will further boost these effects? Perhaps. 

To do: 
Version the existing code and include a new variant with this alternative energy functional. 


"""

def l_opt(kA,kP,A0,P0):
    Sqrt = np.sqrt
    return (6 * 2 ** 0.3333333333333333 * (Sqrt(3) * A0 * kA - 12 * kP)) / (236196 * kA ** 2 * kP * P0 + Sqrt(
        -459165024 * kA ** 3 * (Sqrt(
            3) * A0 * kA - 12 * kP) ** 3 + 55788550416 * kA ** 4 * kP ** 2 * P0 ** 2)) ** 0.3333333333333333 + (
                236196 * kA ** 2 * kP * P0 + Sqrt(-459165024 * kA ** 3 * (Sqrt(
            3) * A0 * kA - 12 * kP) ** 3 + 55788550416 * kA ** 4 * kP ** 2 * P0 ** 2)) ** 0.3333333333333333 / (
                81. * 2 ** 0.3333333333333333 * kA)

def dl_opt(kA,kP,A0,P0):
    Sqrt = np.sqrt
    return (2*0.6666666666666666**0.3333333333333333*(9*kA**2*kP + (81*Sqrt(3)*kA**4*kP**2*P0)/Sqrt(kA**3*(-2*(Sqrt(3)*A0*kA - 12*kP)**3 + 243*kA*kP**2*P0**2))))/(9.*kA*(9*kA**2*kP*P0 + Sqrt(kA**3*(-2*(Sqrt(3)*A0*kA - 12*kP)**3 + 243*kA*kP**2*P0**2))/Sqrt(3))**0.6666666666666666) - ((9*kA**2*kP + (81*Sqrt(3)*kA**4*kP**2*P0)/Sqrt(kA**3*(-2*(Sqrt(3)*A0*kA - 12*kP)**3 + 243*kA*kP**2*P0**2)))*(2**0.6666666666666666*Sqrt(3)*A0*kA**2 - 12*2**0.6666666666666666*kA*kP + 6**0.3333333333333333*(9*kA**2*kP*P0 + Sqrt(kA**3*(-2*(Sqrt(3)*A0*kA - 12*kP)**3 + 243*kA*kP**2*P0**2))/Sqrt(3))**0.6666666666666666))/(9.*3**0.6666666666666666*kA*(9*kA**2*kP*P0 + Sqrt(kA**3*(-2*(Sqrt(3)*A0*kA - 12*kP)**3 + 243*kA*kP**2*P0**2))/Sqrt(3))**1.3333333333333333)



P0_range = np.linspace(1,10)
n_iter = 100
kA = 10**np.random.uniform(-3,1,n_iter)
kP = 10**np.random.uniform(-3,1,n_iter)
A0 = np.random.uniform(1,10,n_iter)

l_vals = np.array([dl_opt(kA,kP,A0,P0) for P0 in P0_range])

plt.plot(l_vals)
plt.show()


@jit
def get_E_A(A,A0):
    return (A-A0)**2 * (A<A0) + (A-A0)*(A>A0)


@jit
def get_E_hexagon(l,A0,P0,gamma,Te):
    A = 3*jnp.sqrt(3)/2 * l**2
    P = 6*jnp.abs(l)
    return 0.02*get_E_A(A,A0) + gamma*(P-P0)**2 + Te*A
    # return 0.02*(A-A0)**2 + gamma*(P-P0)**2 + Te*A + adh*P/A



l_range = np.linspace(4,15,200)
l_vals = np.zeros((20))
fig, ax = plt.subplots()
for i, P0 in enumerate(np.linspace(3,50,20)):
    args = (50,P0,10**-2.7,-0.025)

    e = get_E_hexagon(l_range,*args)
    l_vals[i] = l_range[np.argmin(e)]

    e0 = e.min()#get_E_hexagon(np.sqrt(50/(3*jnp.sqrt(3)/2)), *args)
    ax.plot(l_range,e,color=plt.cm.plasma(i/20),zorder=-i,label="%.1f"%(50-P0))
    ax.scatter(l_range[np.argmin(e)],e[np.argmin(e)],color=plt.cm.plasma(i/20))
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(title="Tension",loc='center left', bbox_to_anchor=(1, 0.5))


ax.set(ylim=(-2.2,0.3),xlim=(3,12),xlabel=r"$\ell$",ylabel=r"$\mathcal{E}$")
# ax.legend()
# plt.plot(l_range,get_E_hexagon(l_range,50,30,0.03,-0.03))
plt.show()

fig, ax = plt.subplots()
ax.plot(np.linspace(0,1,20),l_vals)
ax.set(xlabel=r"$p_{Notch}$",ylabel=r"$\ell^*$")
fig.show()