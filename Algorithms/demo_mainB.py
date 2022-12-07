import hdf5storage
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from skimage.metrics import structural_similarity as ssim

from Algorithms.Function import random_sampling, ShotAlgorithms, PSNR

# ----------------- --------------------
data_name = '../Desarrollo/ReDS/data/cube4.npy'
data_format = 'numpy'

if data_format == 'matlab':
    x = hdf5storage.loadmat(data_name)['data']
else:
    x = np.load(data_name)

if 'data.npy' in data_name:
    x = x.T

'''
---------------  SAMPLING --------------------
'''
sr_rand = 0.5  # 1-compression
_, _, H = random_sampling(x[:, int(x.shape[1] / 2), :], sr_rand, seed=0)

'''
---------------- RECOVERY ALGORITHM -----------------
Select the Algorithm: FISTA , GAP , TWIST , ADMM
'''
case = 'FastMarching'
alg = ShotAlgorithms(x, H)
x_result, hist = alg.get_results(case)
x = x.astype('uint8')

# -------------- Visualization ----------------

pattern_rand = [int(h) for h in H]
pattern_rand = np.array(pattern_rand)

y_rand = x[:, int(x.shape[1] / 2)].copy()
y_rand[:, pattern_rand == 0] = 0

# plt.imshow(y_rand, cmap='gray', aspect='auto')
# plt.title("Removed shots")
# plt.xlabel("Shots")
# plt.ylabel("Time")
# plt.show()

# x = Alg.x
matplotlib.rcParams.update({'font.size': 8})
fig, axs = plt.subplots(2, 4, dpi=250)
fig.suptitle('Results from the ' + str(case) + ' Algorithm')
rem_shots = np.arange(len(pattern_rand))
rem_shots = rem_shots[pattern_rand == 0]

# psnr_vec = []
# for s in rem_shots:
#     psnr_vec.append(PSNR(x[..., s], x_result[..., s]))
# idxs = (-np.array(psnr_vec)).argsort()
# rem_shots = rem_shots[idxs]

# axs[0, 0].imshow(x[..., rem_shots[0]], cmap='gray', aspect='auto')
# axs[0, 0].set_title(f'Reference, shot {rem_shots[0]}')

axs[0, 0].imshow(x[:, int(x.shape[1] / 2)].copy(), cmap='gray', aspect='auto')
axs[0, 0].set_title("Reference")
axs[0, 0].set_xlabel("Shots")
axs[0, 0].set_ylabel("Time")

axs[1, 0].imshow(x[..., rem_shots[1]], cmap='gray', aspect='auto')
axs[1, 0].set_title(f'Reference, shot {rem_shots[1]}')

axs[0, 1].imshow(y_rand, cmap='gray', aspect='auto')
axs[0, 1].set_title("Removed shots")
axs[0, 1].set_xlabel("Shots")
axs[0, 1].set_ylabel("Time")

# metric = PSNR(x[..., rem_shots[0]], x_result[..., rem_shots[0]])
# metric_ssim = ssim(x[..., rem_shots[0]], x_result[..., rem_shots[0]])
# axs[0, 1].imshow(x_result[..., rem_shots[0]], cmap='gray', aspect='auto')
# axs[0, 1].set_title(f'Reconstructed shot {rem_shots[0]}, \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

metric = PSNR(x[..., rem_shots[1]], x_result[..., rem_shots[1]])
metric_ssim = ssim(x[..., rem_shots[1]], x_result[..., rem_shots[1]])
axs[1, 1].imshow(x_result[..., rem_shots[1]], cmap='gray', aspect='auto')
axs[1, 1].set_title(f'Reconstructed shot {rem_shots[1]}, \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

# ====
axs[0, 2].imshow(x[..., rem_shots[2]], cmap='gray', aspect='auto')
axs[0, 2].set_title(f'Reference, shot {rem_shots[2]}')

axs[1, 2].imshow(x[..., rem_shots[3]], cmap='gray', aspect='auto')
axs[1, 2].set_title(f'Reference, shot {rem_shots[3]}')

metric = PSNR(x[..., rem_shots[2]], x_result[..., rem_shots[2]])
metric_ssim = ssim(x[..., rem_shots[2]], x_result[..., rem_shots[2]])
axs[0, 3].imshow(x_result[..., rem_shots[2]], cmap='gray', aspect='auto')
axs[0, 3].set_title(f'Reconstructed shot {rem_shots[2]}, \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

metric = PSNR(x[..., rem_shots[3]], x_result[..., rem_shots[3]])
metric_ssim = ssim(x[..., rem_shots[3]], x_result[..., rem_shots[3]])
axs[1, 3].imshow(x_result[..., rem_shots[3]], cmap='gray', aspect='auto')
axs[1, 3].set_title(f'Reconstructed shot {rem_shots[3]}, \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

fig.tight_layout()
plt.show()

# performance

iteracion = np.linspace(1, len(hist) - 1, len(hist) - 1)

fig = plt.figure()
axes_1 = fig.add_subplot(111)
axes_2 = axes_1.twinx()

color = 'tab:red'
axes_1.set_xlabel('iteraciones')
axes_1.plot(iteracion, hist[1:, 1], color=color, label='SSIM')
axes_1.tick_params(axis='y', labelcolor=color, length=5)
axes_1.grid(axis='both', linestyle='--')

axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

color = 'tab:blue'
axes_2.plot(iteracion, hist[1:, 0], '--', color=color)
axes_1.plot(np.nan, '--', color=color, label='PSNR')
axes_2.tick_params(axis='y', labelcolor=color, length=5)
axes_2.grid(axis='both', linestyle='--')

axes_2.set_yticks(np.linspace(axes_2.get_ybound()[0], axes_2.get_ybound()[1], 8))

# ---------------------------------------

color = 'tab:red'
axes_1.set_xlabel('iteraciones')
axes_1.set_ylabel('ssim', color=color)
axes_1.plot(iteracion, hist[1:, 1], color=color)
axes_1.tick_params(axis='y', labelcolor=color, length=5)
axes_1.yaxis.set_major_locator(MaxNLocator(8))
# axes_1.set_xscale('log')
axes_1.grid(axis='both', which="both", linestyle='--')
axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

color = 'tab:blue'
axes_2.set_ylabel('psnr', color=color)
axes_2.plot(iteracion, hist[1:, 0], color=color)
axes_2.tick_params(axis='y', labelcolor=color, length=5)
axes_2.yaxis.set_major_locator(MaxNLocator(8))
# axes_2.set_xscale('log')
axes_2.grid(axis='both', which="both", linestyle='--')
axes_2.set_yticks(np.linspace(axes_2.get_ybound()[0], axes_2.get_ybound()[1], 8))

plt.show()

print('Fin')
