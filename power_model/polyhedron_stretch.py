import numpy as np
from jax import jit, jacrev,hessian
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
import jax
from scipy import sparse
import triangle as tr
import math

def regular_hexagon_coordinates(center_x, center_y, radius):
    # Initialize a list to store the coordinates of the vertices
    vertices = []

    # Calculate the coordinates of each vertex
    for i in range(6):
        angle_rad = math.radians(60 * i)  # Angle in radians
        x = center_x + radius * math.cos(angle_rad)
        y = center_y + radius * math.sin(angle_rad)
        vertices.append((x, y))

    return vertices

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)


x_basal = np.array([[0,0],
              [0,1.],
              [1.,0],
              [1,1]])

x_basal = np.array(regular_hexagon_coordinates(0,0,1))



theta = 0.5
rotation_vector = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])

x = 2*(x_basal@rotation_vector) + 0.2


norm = np.array([0.3,0.3,1])
norm /= np.linalg.norm(norm)
d = 0.8

fig, ax = plt.subplots()
ax.scatter(*x.T)
fig.show()


##Plane equation: norm[0]x + norm[1]y + norm[2]z -d
##--> z = (d - (norm[0]x + norm[1]y))/norm[2]

# tris = np.array([[0,2,3],[0,3,1]])
tris = tr.triangulate({"vertices":x_basal})["triangles"]

for t in tris:
    _t_extended = np.concatenate([t,(t[0],)])
    plt.plot(*x[_t_extended].T)

"""
Volume of a slanted cuboid 
= 2*3 tetrahedrons 

"""


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

@jit
def get_V_prism(L,norm):
    z_proj = (d-jnp.dot(x,norm[:2]))/norm[2]
    _X = jnp.column_stack([x*L,z_proj])
    tX = _X[tris]
    return slanted_prism_volume(tX[0])

norm = np.array([0.3,0.3,1])
norm /= np.linalg.norm(norm)
d = 0.8
L_range = np.linspace(0,1)
plt.plot(L_range,np.array([jacrev(get_V_prism)(L,norm) for L in L_range]))
norm = np.array([0.0,0.,1])
norm /= np.linalg.norm(norm)
d = 0.8
L_range = np.linspace(0,1)
plt.plot(L_range,np.array([jacrev(get_V_prism)(L,norm) for L in L_range]))

plt.show()

"""
OK so it is a gradient, but it's not flat; and it's dependent on the slope. 
It is however analytically tractable. 
Run the maths, probably best to use Mathematica for this. 

"""

@jit
def get_E_V(L):
    return (get_V_prism(L)-1)**2


@jit
def slanted_cuboid(L,x,norm,d,tris):
    # X = x*L
    z_proj = (d-jnp.dot(x,norm[:2]))/norm[2]
    _X = jnp.column_stack([x*L,z_proj])
    tX = _X[tris]
    A_apical = (0.5*jnp.cross(tX[:,1]-tX[:,0],tX[:,2]-tX[:,0],axis=1)[:,-1]).sum()
    V = slanted_prism_volume(tX[0])+slanted_prism_volume(tX[1])
    return jnp.array([A_apical,V])





L_range = np.linspace(0.1,20)

tx = x[tris]

A = assemble_scalar((np.expand_dims(0.5*np.cross(tx[:,1]-tx[:,0],tx[:,2]-tx[:,0]),1)*np.ones_like(tris)),tris,1)

L = 1

d = 0.5
A_apical, V_true = slanted_cuboid(L,x,norm,d,tris)
A_apical_0, V_0 = slanted_cuboid(L,x,norm,0,tris) #+ A_apical*d


vals = np.array([slanted_cuboid(L,x,norm,d,tris) for L in L_range])

d_range = np.linspace(0,10)
vals = np.array([slanted_cuboid(L,x,norm,d,tris) for d in d_range])

print(stats.linregress(d_range,vals[:,1]))

plt.plot(vals[:,1])
# plt.plot(vals[:,0]/L_range**2)

plt.show()

z_proj = (d - jnp.dot(x, norm[:2])) / norm[2]
_x = jnp.column_stack([x , z_proj])


@jit
def slanted_cuboid2(L,_x,tris):
    # X = x*L
    X = _x.copy()
    _X = X.at[:,:2].multiply(L)
    tX = _X[tris]
    A_vec = (0.5*jnp.cross(tX[:,1]-tX[:,0],tX[:,2]-tX[:,0],axis=1))
    A_apical = A_vec[:,-1].sum()
    A_basal = jnp.linalg.norm(A_vec,axis=1).sum()
    V = slanted_prism_volume(tX[0])+slanted_prism_volume(tX[1])
    return jnp.array([A_apical,V,A_basal])

def assemble_scalar(tval, tri, nc):
    val = jnp.bincount(tri.ravel() + 1, weights=tval.ravel(), length=nc + 1)
    val = val[1:]
    # val = jnp.zeros(nc)
    # for i in range(3):
    #     val = val.at[tri[..., i]].add(tval[..., i])
    return val


L_range = np.linspace(0.1,1.1)

vals = np.array([slanted_cuboid2(L,_x,tris) for L in L_range])

plt.plot(L_range**2,vals[:,1])
plt.plot(L_range**2,vals[:,0])
plt.plot(L_range**2,vals[:,2])
plt.show()

print(stats.linregress(L_range**2,vals[:,0]))

"""
So for a square, regardless of rotation or translation, a stretch in the x,y is effectively

- Note that while volume scales with L^2, the scale factor depends on the geometry. 
- But it is a linear model such that V = alpha*L^2 + no intercept. 



Also note that upon stretch in x,y, angles are preserved I believe. But do check this. 

Next step, out of paranoia, calculate this using a hexagonal projection. 
- Try also by including a small deviation in vertex positions from the major plane. 
"""

