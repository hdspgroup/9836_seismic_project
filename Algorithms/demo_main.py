import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.ticker import MaxNLocator

from Function import *
from skimage.metrics import structural_similarity as ssim

import scipy

# ----------------- --------------------
# x = np.load('../Desarrollo/ReDS/data/data.npy')
data_name = 'data.npy'
x = np.load('../Desarrollo/ReDS/data/' + data_name)
if len(x.shape) > 2:
    x = x[:, :, int(x.shape[-1] / 2)]

if data_name == 'data.npy':
    x = x.T
x = x / np.abs(x).max()
max_itr = 100

'''
---------------  SAMPLING --------------------
'''
sr_rand = 0.4  # 1-compression
y_rand, pattern_rand, pattern_index = random_sampling(x, sr_rand, seed=0)
H = pattern_index

'''
---------------- RECOVERY ALGORITHM -----------------
Select the Algorithm: FISTA , GAP , TWIST , ADMM, DEEPNetwork
'''
case = 'DeepNetwork'
alg = Algorithms(x, H, 'DCT2D', 'IDCT2D')

parameters = {}
if case == 'FISTA':
    parameters = {'max_itr': max_itr,
                  'lmb': 2.91,
                  'mu': 0.39
                  }

elif case == 'GAP':
    parameters = {'max_itr': max_itr,
                  'lmb': 1e-0
                  }

elif case == 'TwIST':
    parameters = {'max_itr': max_itr,
                  'lmb': 0.9,
                  'alpha': 1.2,
                  'beta': 1.998
                  }

elif case == 'ADMM':
    parameters = {'max_itr': max_itr,
                  'lmb': 1e-4,
                  'rho': 1,
                  'gamma': 1
                  }

elif case == 'DeepNetwork':
    parameters = {}

x_result, hist = alg.get_results(case, **parameters)

# -------------- Visualization ----------------

temp = np.asarray(range(0, pattern_rand.shape[0]))
pattern_rand_b2 = np.asarray(pattern_rand, dtype=bool) == 0
H_elim = temp[pattern_rand_b2]

# x = Alg.x

fig, axs = plt.subplots(2, 2, dpi=150)
fig.suptitle('Results from the ' + case + ' Algorithm')

axs[0, 0].imshow(x, cmap='gray', aspect='auto')
axs[0, 0].set_title('Reference')

ytemp = y_rand.copy()
ytemp[:, H_elim] = None
axs[1, 0].imshow(ytemp, cmap='gray', aspect='auto')
axs[1, 0].set_title('Measurements')

# axs[1, 0].sharex(axs[0, 0])
metric = PSNR(x[:, H_elim], x_result[:, H_elim])
metric_ssim = ssim(x[:, H_elim], x_result[:, H_elim], data_range=2.0)
axs[0, 1].imshow(x_result, cmap='gray', aspect='auto')
axs[0, 1].set_title(f'Reconstructed \n PSNR: {metric:0.2f} dB, SSIM:{metric_ssim:0.2f}')
print(metric_ssim)
index = 5
axs[1, 1].plot(x[:, H_elim[index]], 'r', label='Reference')
axs[1, 1].plot(x_result[:, H_elim[index]], 'b', label='Recovered')
axs[1, 1].legend(loc='best')
plt.title('Trace ' + str("{:.0f}".format(H_elim[index])))
axs[1, 1].grid(axis='both', linestyle='--')

fig.tight_layout()
plt.show()

# performance

iteracion = np.linspace(1, len(hist) - 1, len(hist) - 1)

fig = plt.figure()
axes_1 = fig.add_subplot(111)
axes_2 = axes_1.twinx()

color = 'tab:red'
axes_1.set_xlabel('iteraciones')
axes_1.set_ylabel('ssim', color=color)
axes_1.plot(iteracion, hist[1:, 2], color=color)
axes_1.tick_params(axis='y', labelcolor=color, length=5)
axes_1.yaxis.set_major_locator(MaxNLocator(8))
axes_1.set_xscale('log')
axes_1.grid(axis='both', which="both", linestyle='--')
axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

color = 'tab:blue'
axes_2.set_ylabel('psnr', color=color)
axes_2.plot(iteracion, hist[1:, 1], color=color)
axes_2.tick_params(axis='y', labelcolor=color, length=5)
axes_2.yaxis.set_major_locator(MaxNLocator(8))
axes_2.set_xscale('log')
axes_2.grid(axis='both', which="both", linestyle='--')
axes_2.set_yticks(np.linspace(axes_2.get_ybound()[0], axes_2.get_ybound()[1], 8))

plt.show()

print('Fin')
