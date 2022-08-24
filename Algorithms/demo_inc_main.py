import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.ticker import MaxNLocator

from Algorithms.fk_domain import fk
from Algorithms.tv_norm import tv_norm
from Function import *
from skimage.metrics import structural_similarity as ssim

import scipy

# ----------------- --------------------
# x = np.load('../Desarrollo/ReDS/data/data.npy')
data_name = 'inc_data.npy'
x = np.load('../Desarrollo/ReDS/data/incomplete_samples/' + data_name)
if len(x.shape) > 2:
    x = x[:, :, int(x.shape[-1] / 2)]

if data_name == 'data.npy':
    x = x.T

x = np.nan_to_num(x, nan=0)
x = x / np.abs(x).max()
max_itr = 500

'''
---------------  SAMPLING --------------------
'''
# sr_rand = 0.5  # 1-compression
# y_rand, pattern_rand, pattern_index = random_sampling(x, sr_rand, seed=0)
# H = pattern_index
# sr_rand = 0.5  # 1-compression
# y_rand, pattern_rand, pattern_index = random_sampling(x, sr_rand, seed=0)
H = None

'''
---------------- RECOVERY ALGORITHM -----------------
Select the Algorithm: FISTA , GAP , TWIST , ADMM
'''
case = 'FISTA'
alg = Algorithms(x, H, 'DCT2D', 'IDCT2D')
H = alg.H_raw
pattern_rand = np.double(H)
y_rand = x

parameters = {}
if case == 'FISTA':
    parameters = {'max_itr': max_itr,
                  'lmb': 0.1,
                  'mu': 0.3
                  }

elif case == 'GAP':
    parameters = {'max_itr': max_itr,
                  'lmb': 1e-0
                  }

elif case == 'TwIST':
    parameters = {'max_itr': max_itr,
                  'lmb': 0.5,
                  'alpha': 1.2,
                  'beta': 1.998
                  }

elif case == 'ADMM':
    parameters = {'max_itr': max_itr,
                  'lmb': 5e-4,
                  'rho': 1,
                  'gamma': 1
                  }

x_result, hist = alg.get_results(case, **parameters)

# -------------- Visualization ----------------

temp = np.asarray(range(0, pattern_rand.shape[0]))
pattern_rand_b2 = np.asarray(pattern_rand, dtype=bool) == 0
H_elim = temp[pattern_rand_b2]

# x = Alg.x

fig, axs = plt.subplots(2, 3, dpi=150, figsize=(12, 8))
fig.suptitle('Resultados del algoritmo ' + case)

metric = tv_norm(x)
axs[0, 0].imshow(x, cmap='gray', aspect='auto')
axs[0, 0].set_title(f'Reference \n TV-norm: {metric:0.2f}')

metric = tv_norm(x_result)
axs[0, 1].imshow(x_result, cmap='gray', aspect='auto')
axs[0, 1].set_title(f'Reconstructed \n TV norm: {metric:0.2f}')

index = 5
axs[0, 2].plot(x_result[:, H_elim[index]], 'b', label='Recovered')
axs[0, 2].legend(loc='best')
axs[0, 2].set_title('Trace ' + str("{:.0f}".format(H_elim[index])))
axs[0, 2].grid(axis='both', linestyle='--')

# fk domain

# calculate FK domain
dt = 0.568
dx = 5
FK, f, kx = fk(x, dt, dx)

axs[1, 0].imshow(FK[:200], aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
axs[1, 0].set_title('FK reference')

FK, f, kx = fk(x_result, dt, dx)

axs[1, 1].imshow(FK[:200], aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
axs[1, 1].set_title('FK reconstruction')

index = -1
axs[1, 2].plot(x_result[:, H_elim[index]], 'b', label='Recovered')
axs[1, 2].legend(loc='best')
axs[1, 2].set_title('Trace ' + str("{:.0f}".format(H_elim[index])))
axs[1, 2].grid(axis='both', linestyle='--')

plt.show()

# performance

iteracion = np.linspace(1, len(hist) - 1, len(hist) - 1)

fig = plt.figure()
fig.suptitle('Resultados del experimento')

axes_1 = fig.add_subplot(111)
axes_2 = axes_1.twinx()

color = 'tab:red'
axes_1.set_xlabel('iteraciones')
axes_1.plot(iteracion, hist[1:, 0], color=color, label='Error residual')
axes_1.tick_params(axis='y', labelcolor=color, length=5)
axes_1.yaxis.set_major_locator(MaxNLocator(8))
axes_1.grid(axis='both', which="both", linestyle='--')
axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

color = 'tab:blue'
axes_2.plot(iteracion, hist[1:, 3], '--', color=color)
axes_1.plot(np.nan, '--', color=color, label='Norma TV')
axes_2.tick_params(axis='y', labelcolor=color, length=5)
axes_2.yaxis.set_major_locator(MaxNLocator(8))
axes_2.grid(axis='both', which="both", linestyle='--')
axes_2.set_yticks(np.linspace(axes_2.get_ybound()[0], axes_2.get_ybound()[1], 8))

axes_1.legend(loc='best')
plt.show()

print('Fin')
