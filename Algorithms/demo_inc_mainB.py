import hdf5storage
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from skimage.metrics import structural_similarity as ssim

from Algorithms.Function import random_sampling, ShotAlgorithms, PSNR
from Algorithms.fk_domain import fk
from Algorithms.tv_norm import tv_norm

# data = alg.measurements().permute(0, 2, 1)
# np.save('inc_cube4.npy', data)

# ----------------- --------------------
data_name = '../Desarrollo/ReDS/data/incomplete_samples/inc_cube4.npy'
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
# sr_rand = 0.5  # 1-compression
# _, _, H = random_sampling(x[:, int(x.shape[1] / 2), :], sr_rand, seed=0)
H = None

'''
---------------- RECOVERY ALGORITHM -----------------
Select the Algorithm: FISTA , GAP , TWIST , ADMM
'''
case = 'FastMarching'
alg = ShotAlgorithms(x, H)
x_result, hist = alg.get_results(case)
x = x.astype('uint8')

H = alg.H

# -------------- Visualization ----------------

pattern_rand = [int(h) for h in H]
pattern_rand = np.array(pattern_rand)
y_rand = x[:, int(x.shape[1] / 2)].copy()

# plt.imshow(y_rand, cmap='gray', aspect='auto')
# plt.title("Removed shots")
# plt.xlabel("Shots")
# plt.ylabel("Time")
# plt.show()

dt = 0.568
dx = 5

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

axs[0, 0].imshow(x[:, int(x.shape[1] / 2)].copy(), cmap='gray', aspect='auto')
axs[0, 0].set_title("Reference")
axs[0, 0].set_xlabel("Shots")
axs[0, 0].set_ylabel("Time")

FK, f, kx = fk(x_result[..., rem_shots[1]], dt, dx)
axs[1, 0].imshow(FK, aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
axs[1, 0].set_title(f'FK reference, shot {rem_shots[1]}')

axs[0, 1].imshow(x_result[:, int(x.shape[1] / 2)].copy(), cmap='gray', aspect='auto')
axs[0, 1].set_title("Reconstructed shots")
axs[0, 1].set_xlabel("Shots")
axs[0, 1].set_ylabel("Time")

metric = tv_norm(x_result[..., rem_shots[1]])
axs[1, 1].imshow(x_result[..., rem_shots[1]], cmap='gray', aspect='auto')
axs[1, 1].set_title(f'Reconstructed shot {rem_shots[1]}, \n Norma TV/L1: {metric:0.4f}')

# ====
FK, f, kx = fk(x_result[..., rem_shots[2]], dt, dx)
axs[0, 2].imshow(FK, aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
axs[0, 2].set_title(f'FK Reference, shot {rem_shots[2]}')

FK, f, kx = fk(x_result[..., rem_shots[3]], dt, dx)
axs[1, 2].imshow(FK, aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
axs[1, 2].set_title(f'FK Reference, shot {rem_shots[3]}')

metric = tv_norm(x_result[..., rem_shots[2]])
axs[0, 3].imshow(x_result[..., rem_shots[2]], cmap='gray', aspect='auto')
axs[0, 3].set_title(f'Reconstructed shot {rem_shots[2]}, \n Norma TV/L1: {metric:0.4f}')

metric = tv_norm(x_result[..., rem_shots[3]])
axs[1, 3].imshow(x_result[..., rem_shots[3]], cmap='gray', aspect='auto')
axs[1, 3].set_title(f'Reconstructed shot {rem_shots[3]}, \n Norma TV/L1: {metric:0.4f}')

fig.tight_layout()
plt.show()

# performance

iteracion = np.linspace(1, len(hist) - 1, len(hist) - 1)

fig = plt.figure()
axes = fig.add_subplot(111)

color = 'tab:blue'
axes.set_xlabel('iteraciones')
axes.plot(iteracion, hist[1:, -1], color=color, label='Norma TV/L1')
axes.tick_params(axis='y', labelcolor=color, length=5)
axes.grid(axis='both', linestyle='--')

axes.set_yticks(np.linspace(axes.get_ybound()[0], axes.get_ybound()[1], 8))
axes.legend(loc='best')

plt.show()

print('Fin')
