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