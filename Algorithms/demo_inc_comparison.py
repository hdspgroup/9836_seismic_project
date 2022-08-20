import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.ticker import MaxNLocator

from Function import *
from skimage.metrics import structural_similarity as ssim

import scipy

# ----------------- --------------------
# x = np.load('../data/data.npy')
data_name = 'inc_data.npy'
x = np.load('../data/' + data_name)
if len(x.shape) > 2:
    x = x[:, :, int(x.shape[-1] / 2)]

if data_name == 'data.npy':
    x = x.T

x = np.nan_to_num(x, nan=0)
x = x / np.abs(x).max()
max_itr = 100

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
case = 'GAP'
alg = Algorithms(x, H, 'DCT2D', 'IDCT2D')
H = alg.H_raw
pattern_rand = np.double(H)
y_rand = x


fista_parameters = {'max_itr': max_itr,
                    'lmb': 2.9,
                    'mu': 0.4
                    }

gap_parameters = {'max_itr': max_itr,
                  'lmb': 30.0
                  }

twist_parameters = {'max_itr': max_itr,
                    'lmb': 17.0,
                    'alpha': 1.2,
                    'beta': 1.998
                    }

admm_parameters = {'max_itr': max_itr,
                   'lmb': 0.0005,
                   'rho': 0.5,
                   'gamma': 1.05
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

fig, axs = plt.subplots(2, 3, dpi=300, figsize=(12, 9))
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

metric = PSNR(x[:, H_elim], x_result_twist[:, H_elim])
metric_ssim = ssim(x[:, H_elim], x_result_twist[:, H_elim])
axs[1, 1].imshow(x_result_twist, cmap='gray', aspect='auto')
axs[1, 1].set_title(f'TwIST\nPSNR: {metric:0.2f} dB, SSIM:{metric_ssim:0.2f}')

metric = PSNR(x[:, H_elim], x_result_gap[:, H_elim])
metric_ssim = ssim(x[:, H_elim], x_result_gap[:, H_elim])
axs[0, 2].imshow(x_result_gap, cmap='gray', aspect='auto')
axs[0, 2].set_title(f'GAP\nPSNR: {metric:0.2f} dB, SSIM:{metric_ssim:0.2f}')

metric = PSNR(x[:, H_elim], x_result_admm[:, H_elim])
metric_ssim = ssim(x[:, H_elim], x_result_admm[:, H_elim])
axs[1, 2].imshow(x_result_admm, cmap='gray', aspect='auto')
axs[1, 2].set_title(f'ADMM\nPSNR: {metric:0.2f} dB, SSIM:{metric_ssim:0.2f}')

plt.show()


# performance

iteracion = np.linspace(1, len(hist_fista), len(hist_fista))

figure = plt.figure()
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92, wspace=0.5, hspace=0.3)
axs = figure.subplots(2, 2)

# fig = plt.figure()
axes_1 = axs[0, 0]
axes_2 = axes_1.twinx()

color = 'tab:red'
axes_1.set_xlabel('iteraciones')
axes_1.set_ylabel('ssim', color=color)
axes_1.plot(iteracion[1:], hist_fista[1:, 2], color=color)
axes_1.set_title('FISTA')
axes_1.tick_params(axis='y', labelcolor=color, length=5)
axes_1.grid(axis='both', linestyle='--')

axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

color = 'tab:blue'
axes_2.set_ylabel('psnr', color=color)
axes_2.plot(iteracion[1:], hist_fista[1:, 1], color=color)
axes_2.tick_params(axis='y', labelcolor=color, length=5)
axes_1.grid(axis='both', linestyle='--')

axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

axes_1 = axs[0, 1]
axes_2 = axes_1.twinx()

color = 'tab:red'
axes_1.set_xlabel('iteraciones')
axes_1.set_ylabel('ssim', color=color)
axes_1.plot(iteracion[1:], hist_gap[1:, 2], color=color)
axes_1.set_title('GAP')
axes_1.tick_params(axis='y', labelcolor=color, length=5)
axes_1.grid(axis='both', linestyle='--')

axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

color = 'tab:blue'
axes_2.set_ylabel('psnr', color=color)
axes_2.plot(iteracion[1:], hist_gap[1:, 1], color=color)
axes_2.tick_params(axis='y', labelcolor=color, length=5)
axes_1.grid(axis='both', linestyle='--')

axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

axes_1 = axs[1, 0]
axes_2 = axes_1.twinx()

color = 'tab:red'
axes_1.set_xlabel('iteraciones')
axes_1.set_ylabel('ssim', color=color)
axes_1.plot(iteracion[1:], hist_twist[1:, 2], color=color)
axes_1.set_title('TwIST')
axes_1.tick_params(axis='y', labelcolor=color, length=5)
axes_1.grid(axis='both', linestyle='--')

axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

color = 'tab:blue'
axes_2.set_ylabel('psnr', color=color)
axes_2.plot(iteracion[1:], hist_twist[1:, 1], color=color)
axes_2.tick_params(axis='y', labelcolor=color, length=5)
axes_1.grid(axis='both', linestyle='--')

axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

axes_1 = axs[1, 1]
axes_2 = axes_1.twinx()

color = 'tab:red'
axes_1.set_xlabel('iteraciones')
axes_1.set_ylabel('ssim', color=color)
axes_1.plot(iteracion[1:], hist_admm[1:, 2], color=color)
axes_1.set_title('ADMM')
axes_1.tick_params(axis='y', labelcolor=color, length=5)
axes_1.grid(axis='both', linestyle='--')

axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

color = 'tab:blue'
axes_2.set_ylabel('psnr', color=color)
axes_2.plot(iteracion[1:], hist_admm[1:, 1], color=color)
axes_2.tick_params(axis='y', labelcolor=color, length=5)
axes_1.grid(axis='both', linestyle='--')

axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

plt.show()

print('Fin')
