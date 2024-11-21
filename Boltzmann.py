#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from matplotlib.animation import PillowWriter
import scienceplots
plt.style.use(['science', 'notebook'])
from itertools import combinations

#%%
n_particles = 400
r = np.random.random((2, n_particles))

#%%
ixr = r[0] > 0.5
ixl = r[0] <= 0.5
#%%
# assign IDs to particles
ids = np.arange(n_particles)

# plot initial configuration of particles
plt.figure(figsize=(5,5))
plt.scatter(r[0][ixr], r[1][ixr], color = 'r', s=6)
plt.scatter(r[0][ixl], r[1][ixl], color='b', s=6)

# %%
v= np.zeros((2, n_particles))
v[0][ixr] = -500
v[0][ixl] = 500

# %%
listOfPairs = list(combinations(ids,2))
pairsArray = np.asarray(listOfPairs)

# %%
x_pairs = np.asarray(list(combinations(r[0],2)))
y_pairs = np.asarray(list(combinations(r[1],2)))

# %%
dx_pairs = np.diff(x_pairs, axis = 1).ravel()
dy_pairs = np.diff(y_pairs, axis = 1).ravel()

#%% square of the distance
d_pairs = dx_pairs**2 + dy_pairs**2

# %% get colliding pairs
radius = 0.06
ids_pairs_collide = pairsArray[d_pairs < 4*radius**2]
 # %%
v1 = v[:,ids_pairs_collide[:,0]]
v2 = v[:,ids_pairs_collide[:,1]]
r1 = r[:,ids_pairs_collide[:,0]]
r2 = r[:,ids_pairs_collide[:,1]]

#%%
v1new = v1 - np.diag((v1-v2).T@(r1-r2))/np.sum((r1-r2)**2, axis=0) * (r1-r2)
v2new = v2 - np.diag((v2-v1).T@(r2-r1))/np.sum((r2-r1)**2, axis=0) * (r2-r1)

# %%
def get_delta_pairs(x):
    return np.diff(np.asarray(list(combinations(x,2))), axis=1).ravel()

def get_deltad_pairs(r):
    return (get_delta_pairs(r[0])**2 + get_delta_pairs(r[1])**2)

def compute_new_v(v1, v2, r1, r2):
    v1new = v1 - np.diag((v1-v2).T@(r1-r2))/np.sum((r1-r2)**2, axis=0) * (r1-r2)
    v2new = v2 - np.diag((v2-v1).T@(r2-r1))/np.sum((r2-r1)**2, axis=0) * (r2-r1)
    return v1new, v2new

def motion(r, v, id_pairs, ts, dt, d_cutoff):
    rs = np.zeros((ts, r.shape[0], r.shape[1]))
    vs = np.zeros((ts, v.shape[0], v.shape[1]))
    #initial state
    rs[0] = r.copy()
    vs[0] = v.copy()
    for i in range(1, ts):
        ic = id_pairs[get_deltad_pairs(r) < d_cutoff]
        v[:,ic[:,0]], v[:, ic[:,1]] = compute_new_v(v[:, ic[:,0]], v[:,ic[:,1]], r[:,ic[:,0]], r[:,ic[:,1]])

        v[0,r[0]>1] = -np.abs(v[0,r[0]>1])
        v[0,r[0]<0] = np.abs(v[0,r[0]<0])
        v[1,r[1]>1] = -np.abs(v[1,r[1]>1])
        v[1,r[1]<0] = np.abs(v[1,r[1]<0])

        r = r + v*dt
        rs[i] = r.copy()
        vs[i] = v.copy()
    return rs, vs

#%%
radius = 0.06
rs, vs = motion(r, v, pairsArray, ts=1000, dt=0.000008, d_cutoff=(2*radius)**2 )

# %%
fig, ax = plt.subplots(1, 1, figsize=(5,5))
xred, yred = rs[0][0][ixr], rs[0][1][ixr]
xblue, yblue = rs[0][0][ixl],rs[0][1][ixl]
circles_red = [plt.Circle((xi, yi), radius=radius, linewidth=0) for xi, yi in zip(xred, yred)]
circles_blue = [plt.Circle((xi, yi), radius=radius, linewidth=0) for xi, yi in zip(xblue, yblue)]
cred = matplotlib.collections.PatchCollection(circles_red, facecolors='red')
cblue = matplotlib.collections.PatchCollection(circles_blue, facecolors='blue')
ax.add_collection(cred)
ax.add_collection(cblue)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
plt.show()

#%%
fig, ax = plt.subplots(1,1,figsize=(5,5))

def animate(i):
    ax.clear()
    xred, yred = rs[i][0][ixr], rs[i][1][ixr]
    xblue, yblue = rs[i][0][ixl],rs[i][1][ixl]
    circles_red = [plt.Circle((xi, yi), radius=radius/4, linewidth=0) for xi, yi in zip(xred, yred)]
    circles_blue = [plt.Circle((xi, yi), radius=radius/4, linewidth=0) for xi, yi in zip(xblue, yblue)]
    cred = matplotlib.collections.PatchCollection(circles_red, facecolors='red')
    cblue = matplotlib.collections.PatchCollection(circles_blue, facecolors='blue')
    ax.add_collection(cred)
    ax.add_collection(cblue)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

ani = animation.FuncAnimation(fig, animate, frames=500, interval=50)
ani.save('ani3.gif',writer='pillow',fps=30,dpi=100)
# %%
