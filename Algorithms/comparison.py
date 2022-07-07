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
case = 'FISTA'
alg = Algorithms(x, H, 'DCT2D', 'IDCT2D')

fista_parameters = {'max_itr': max_itr,
                    'lmb': 0.1,
                    'mu': 0.3
                    }

gap_parameters = {'max_itr': max_itr,
                  'lmb': 1e-0
                  }

twist_parameters = {'max_itr': max_itr,
                    'lmb': 0.5,
                    'alpha': 1.2,
                    'beta': 1.998
                    }

admm_parameters = {'max_itr': max_itr,
                   'lmb': 5e-4,
                   'rho': 1,
                   'gamma': 1
                   }

x_result_fista, hist_fista = alg.get_results('FISTA', **fista_parameters)
x_result_gap, hist_gap = alg.get_results('GAP', **gap_parameters)
x_result_twist, hist_twist = alg.get_results('TwIST', **twist_parameters)
x_result_admm, hist_admm = alg.get_results('ADMM', **admm_parameters)

# -------------- Visualization ----------------

temp = np.asarray(range(0, pattern_rand.shape[0]))
pattern_rand_b2 = np.asarray(pattern_rand, dtype=bool) == 0
H_elim = temp[pattern_rand_b2]

# graphics

fig, axs = plt.subplots(2, 3, dpi=150)
# fig.suptitle('Comparaciones')

axs[0, 0].imshow(x, cmap='gray', aspect='auto')
axs[0, 0].set_title('Reference')

ytemp = y_rand.copy()
ytemp[:, H_elim] = None
axs[1, 0].imshow(ytemp, cmap='gray', aspect='auto')
axs[1, 0].set_title('Measurements')

metric = PSNR(x[:, H_elim], x_result_fista[:, H_elim])
metric_ssim = ssim(x[:, H_elim], x_result_fista[:, H_elim])
axs[0, 1].imshow(x_result_fista, cmap='gray', aspect='auto')
axs[0, 1].set_title(f'FISTA\nPSNR: {metric:0.2f} dB, SSIM:{metric_ssim:0.2f}')

metric = PSNR(x[:, H_elim], x_result_gap[:, H_elim])
metric_ssim = ssim(x[:, H_elim], x_result_gap[:, H_elim])
axs[1, 1].imshow(x_result_gap, cmap='gray', aspect='auto')
axs[1, 1].set_title(f'GAP\nPSNR: {metric:0.2f} dB, SSIM:{metric_ssim:0.2f}')

metric = PSNR(x[:, H_elim], x_result_twist[:, H_elim])
metric_ssim = ssim(x[:, H_elim], x_result_twist[:, H_elim])
axs[0, 2].imshow(x_result_twist, cmap='gray', aspect='auto')
axs[0, 2].set_title(f'TwIST\nPSNR: {metric:0.2f} dB, SSIM:{metric_ssim:0.2f}')

metric = PSNR(x[:, H_elim], x_result_admm[:, H_elim])
metric_ssim = ssim(x[:, H_elim], x_result_admm[:, H_elim])
axs[1, 2].imshow(x_result_admm, cmap='gray', aspect='auto')
axs[1, 2].set_title(f'ADMM\nPSNR: {metric:0.2f} dB, SSIM:{metric_ssim:0.2f}')

plt.show()


# performance

iteracion = np.linspace(1, len(hist_fista), len(hist_fista))
fig, axs = plt.subplots(2, 2, dpi=150)

# fig = plt.figure()
axes_1 = axs[0, 0]
axes_2 = axes_1.twinx()

color = 'tab:red'
axes_1.set_xlabel('iteraciones')
axes_1.set_ylabel('ssim', color=color)
axes_1.plot(iteracion, hist_fista[:, 2], color=color)
axes_1.set_title('FISTA')
axes_1.tick_params(axis='y', labelcolor=color, length=5)
axes_1.yaxis.set_major_locator(MaxNLocator(8))

color = 'tab:blue'
axes_2.set_ylabel('psnr', color=color)
axes_2.plot(iteracion, hist_fista[:, 1], color=color)
axes_2.tick_params(axis='y', labelcolor=color, length=5)
axes_2.yaxis.set_major_locator(MaxNLocator(8))

axes_1 = axs[0, 1]
axes_2 = axes_1.twinx()

color = 'tab:red'
axes_1.set_xlabel('iteraciones')
axes_1.set_ylabel('ssim', color=color)
axes_1.plot(iteracion, hist_gap[:, 2], color=color)
axes_1.set_title('GAP')
axes_1.tick_params(axis='y', labelcolor=color, length=5)
axes_1.yaxis.set_major_locator(MaxNLocator(8))

color = 'tab:blue'
axes_2.set_ylabel('psnr', color=color)
axes_2.plot(iteracion, hist_gap[:, 1], color=color)
axes_2.tick_params(axis='y', labelcolor=color, length=5)
axes_2.yaxis.set_major_locator(MaxNLocator(8))

axes_1 = axs[1, 0]
axes_2 = axes_1.twinx()

color = 'tab:red'
axes_1.set_xlabel('iteraciones')
axes_1.set_ylabel('ssim', color=color)
axes_1.plot(iteracion, hist_twist[:, 2], color=color)
axes_1.set_title('TwIST')
axes_1.tick_params(axis='y', labelcolor=color, length=5)
axes_1.yaxis.set_major_locator(MaxNLocator(8))

color = 'tab:blue'
axes_2.set_ylabel('psnr', color=color)
axes_2.plot(iteracion, hist_twist[:, 1], color=color)
axes_2.tick_params(axis='y', labelcolor=color, length=5)
axes_2.yaxis.set_major_locator(MaxNLocator(8))

axes_1 = axs[1, 1]
axes_2 = axes_1.twinx()

color = 'tab:red'
axes_1.set_xlabel('iteraciones')
axes_1.set_ylabel('ssim', color=color)
axes_1.plot(iteracion, hist_admm[:, 2], color=color)
axes_1.set_title('ADMM')
axes_1.tick_params(axis='y', labelcolor=color, length=5)
axes_1.yaxis.set_major_locator(MaxNLocator(8))

color = 'tab:blue'
axes_2.set_ylabel('psnr', color=color)
axes_2.plot(iteracion, hist_admm[:, 1], color=color)
axes_2.tick_params(axis='y', labelcolor=color, length=5)
axes_2.yaxis.set_major_locator(MaxNLocator(8))

plt.show()

print('Fin')
