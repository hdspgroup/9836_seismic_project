from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.ticker import MaxNLocator

from Function import *
from skimage.metrics import structural_similarity as ssim

import scipy

# ----------------- --------------------
# x = np.load('../data/data.npy')
data_name = 'data.npy'
x = np.load('../data/' + data_name)
if len(x.shape) > 2:
    x = x[:, :, int(x.shape[-1] / 2)]

if data_name == 'data.npy':
    x = x.T
x = x / np.abs(x).max()
max_itr = 100

'''
---------------  SAMPLING --------------------
'''
sr_rand = 0.5  # 1-compression
y_rand, pattern_rand, pattern_index = random_sampling(x, sr_rand, seed=0)
H = pattern_index

'''
---------------- RECOVERY ALGORITHM -----------------
Select the Algorithm: FISTA , GAP , TWIST , ADMM
'''
case = 'ADMM'
alg = Algorithms(x, H, 'DCT2D', 'IDCT2D')

# Prueba con el algoritmo GAP
lmbs = np.logspace(1.0, 2.2, 15)

# Prueba con el algoritmo ADMM
rhos = np.logspace(-3, 1.5, 15)

parameters = {}
performances = []

if case == 'GAP':
    for num_run, lmb in enumerate(product(lmbs)):

        parameters = {'max_itr': max_itr,
                      'lmb': lmb[0]
                      }

        x_result, hist = alg.get_results(case, **parameters)
        performances.append(np.concatenate([lmb[0], hist[-1, :]]))

        print(f'Experimento {num_run}')

elif case == 'ADMM':
    for num_run, rho in enumerate(product(rhos)):

        parameters = {'max_itr': max_itr,
                      'lmb': 5e-4,
                      'rho': rho[0],
                      'gamma': 1
                      }

        x_result, hist = alg.get_results(case, **parameters)
        performances.append(np.concatenate([rho, hist[-1, :]]))

        print(f'Experimento {num_run}')

performances = np.array(performances)

# -------------- Performance ----------------

fig = plt.figure()
axes_1 = fig.add_subplot(111)
axes_2 = axes_1.twinx()

color = 'tab:red'
axes_1.set_xlabel(r'$\rho$ | Valores fijos: $\gamma=1.0$ $\lambda=0.0005$')
axes_1.set_ylabel('ssim', color=color)
axes_1.plot(performances[:, 0], performances[:, -1], '--o', color=color)
axes_1.tick_params(axis='y', labelcolor=color, length=5)
axes_1.yaxis.set_major_locator(MaxNLocator(8))
axes_1.set_xscale('log')

color = 'tab:blue'
axes_2.set_ylabel('ssim', color=color)
axes_2.plot(performances[:, 0], performances[:, -2], '--o', color=color)
axes_2.tick_params(axis='y', labelcolor=color, length=5)
axes_2.yaxis.set_major_locator(MaxNLocator(8))
axes_2.set_xscale('log')

# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#



plt.show()
