import cv2
import numpy as np
import matplotlib.pyplot as plt

from Algorithms.tv_norm import *

# ----------------- --------------------
# x = np.load('../Desarrollo/ReDS/data/data.npy')
data_name = 'data.npy'
x = np.load('../Desarrollo/ReDS/data/' + data_name)
if len(x.shape) > 2:
    x = x[:, :, int(x.shape[-1] / 2)]

if data_name == 'data.npy':
    x = x.T
x = x / np.abs(x).max()


# ----------------- --------------------
# x = np.load('../Desarrollo/ReDS/data/data.npy')
data_name = 'inc_data.npy'
x_inc = np.load('../Desarrollo/ReDS/data/incomplete_samples/' + data_name)
if len(x_inc.shape) > 2:
    x_inc = x_inc[:, :, int(x_inc.shape[-1] / 2)]

if data_name == 'data.npy':
    x_inc = x_inc.T

x_inc = np.nan_to_num(x_inc, nan=0)
x_inc = x_inc / np.abs(x_inc).max()
max_itr = 500


fig, axs = plt.subplots(1, 2, dpi=150, figsize=(8, 4))
fig.suptitle('TV norm')

axs[0].imshow(x, cmap='gray', aspect='auto')
axs[0].set_title('Reference')
axs[0].set_xlabel(f'TV norm: {tv_norm(x):.2f}')

axs[1].imshow(x_inc, cmap='gray', aspect='auto')
axs[1].set_title('Measurement')
axs[1].set_xlabel(f'TV norm: {tv_norm(x_inc):.2f}')

plt.show()

print('Fin')
