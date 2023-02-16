import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from Algorithms.fk_domain import fk, FK_visualize

data2 = np.load('../Desarrollo/ReDS/data/data.npy').T
print('data2 dimensions are:', data2.shape)

# calculate FK domain
dt = 0.568
dx = 5
FK, f, kx = fk(data2, dt, dx)

# visualize TX and FK domains
figure(figsize=(5, 5))
FK_visualize(FK, kx, 200)
plt.show()
