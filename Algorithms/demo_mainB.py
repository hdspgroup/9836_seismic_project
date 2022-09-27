import hdf5storage
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
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

plt.imshow(y_rand, cmap='seismic', aspect='auto')
plt.title("Removed shots")
plt.xlabel("Shots")
plt.ylabel("Time")
plt.show()

# x = Alg.x
matplotlib.rcParams.update({'font.size': 8})
fig, axs = plt.subplots(2, 4, dpi=250)
fig.suptitle('Results from the ' + str(case) + ' Algorithm')
rem_shots = np.arange(len(pattern_rand))
rem_shots = rem_shots[pattern_rand == 0]

psnr_vec = []
for s in rem_shots:
    psnr_vec.append(PSNR(x[..., s], x_result[..., s]))
idxs = (-np.array(psnr_vec)).argsort()
rem_shots = rem_shots[idxs]
axs[0, 0].imshow(x[..., rem_shots[0]], cmap='seismic', aspect='auto')
axs[0, 0].set_title(f'Reference, shot {rem_shots[0]}')

axs[1, 0].imshow(x[..., rem_shots[1]], cmap='seismic', aspect='auto')
axs[1, 0].set_title(f'Reference, shot {rem_shots[1]}')

metric = PSNR(x[..., rem_shots[0]], x_result[..., rem_shots[0]])
metric_ssim = ssim(x[..., rem_shots[0]], x_result[..., rem_shots[0]])
axs[0, 1].imshow(x_result[..., rem_shots[0]], cmap='seismic', aspect='auto')
axs[0, 1].set_title(f'Reconstructed shot {rem_shots[0]}, \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

metric = PSNR(x[..., rem_shots[1]], x_result[..., rem_shots[1]])
metric_ssim = ssim(x[..., rem_shots[1]], x_result[..., rem_shots[1]])
axs[1, 1].imshow(x_result[..., rem_shots[1]], cmap='seismic', aspect='auto')
axs[1, 1].set_title(f'Reconstructed shot {rem_shots[1]}, \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

# ====
axs[0, 2].imshow(x[..., rem_shots[2]], cmap='seismic', aspect='auto')
axs[0, 2].set_title(f'Reference, shot {rem_shots[2]}')

axs[1, 2].imshow(x[..., rem_shots[3]], cmap='seismic', aspect='auto')
axs[1, 2].set_title(f'Reference, shot {rem_shots[3]}')

metric = PSNR(x[..., rem_shots[2]], x_result[..., rem_shots[2]])
metric_ssim = ssim(x[..., rem_shots[2]], x_result[..., rem_shots[2]])
axs[0, 3].imshow(x_result[..., rem_shots[2]], cmap='seismic', aspect='auto')
axs[0, 3].set_title(f'Reconstructed shot {rem_shots[2]}, \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

metric = PSNR(x[..., rem_shots[3]], x_result[..., rem_shots[3]])
metric_ssim = ssim(x[..., rem_shots[3]], x_result[..., rem_shots[3]])
axs[1, 3].imshow(x_result[..., rem_shots[3]], cmap='seismic', aspect='auto')
axs[1, 3].set_title(f'Reconstructed shot {rem_shots[3]}, \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

fig.tight_layout()
plt.show()

print('Fin')
