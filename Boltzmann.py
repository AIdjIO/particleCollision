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
