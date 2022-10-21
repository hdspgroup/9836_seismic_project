# Test different metrics given shot samples

import numpy as np
import matplotlib.pyplot as plt

from Algorithms.Function import PSNR

# generate datasets

x = np.load('../Desarrollo/ReDS/data/data.npy').T
x = x / np.abs(x).max()
np.save('../Desarrollo/ReDS/data/shots/data_sample.npy', x)

x = np.load('../Desarrollo/ReDS/data/cube4.npy')
x = x[:, :, int(x.shape[-1] / 2)]
x = x / np.abs(x).max()
np.save('../Desarrollo/ReDS/data/shots/cub4_sample.npy', x)

x = np.load('../Desarrollo/ReDS/data/spii15s.npy')
x = x[:, :, int(x.shape[-1] / 2)]
x = x / np.abs(x).max()
np.save('../Desarrollo/ReDS/data/shots/spii15s_sample.npy', x)

x = np.load('../Desarrollo/ReDS/data/syn3D_cross-spread2.npy')
x = x[:, :, int(x.shape[-1] / 2)]
x = x / np.abs(x).max()
np.save('../Desarrollo/ReDS/data/shots/syn3D_cross-spread2_sample.npy', x)


# -------------------------------------------------------------------------------

def add_noise(image, snr):
    """Add noise to an image"""

    # calculate the signal power
    sig_power = np.sum(image ** 2) / image.size

    # calculate the noise power based on the SNR
    noise_power = sig_power / (10 ** (snr / 10))

    # generate a matrix of random values with the same dimensions as the image
    noise = np.random.normal(0, np.sqrt(noise_power), image.shape)

    # add the noise to the image
    noisy_image = image + noise

    return noisy_image


def tv_norm(image, norm=None):
    """Compute the mean (isotropic) TV norm of an image"""

    grad_x1 = np.diff(image, axis=0)
    grad_x2 = np.diff(image, axis=1)

    performance = np.sqrt(grad_x1[:, :-1] ** 2 + grad_x2[:-1, :] ** 2).sum()

    if norm == 'l0':
        performance /= np.linalg.norm(np.reshape(image, -1), ord=0)
    elif norm == 'l1':
        performance /= np.linalg.norm(image, ord=1)
    elif norm == 'l2':
        performance /= np.linalg.norm(image, ord=2)
    elif norm == 'l21':
        performance /= np.linalg.norm(image, ord=2, axis=1).sum()

    return performance


# load dataset sample

x = np.load('../Desarrollo/ReDS/data/shots/data_sample.npy')
x_noise = add_noise(x, 1)

pnsr_image = PSNR(x, x_noise)

tv_image = tv_norm(x)
tv_image_l0 = tv_norm(x, norm='l0')
tv_image_l1 = tv_norm(x, norm='l1')
tv_image_l2 = tv_norm(x, norm='l2')
tv_image_l21 = tv_norm(x, norm='l21')

tv_noise_image = tv_norm(x_noise)
tv_noise_image_l0 = tv_norm(x_noise, norm='l0')
tv_noise_image_l1 = tv_norm(x_noise, norm='l1')
tv_noise_image_l2 = tv_norm(x_noise, norm='l2')
tv_noise_image_l21 = tv_norm(x, norm='l21')

print('PSNR: ', pnsr_image)
print('TV: ', tv_image)
print('TV L0: ', tv_image_l0)
print('TV L1: ', tv_image_l1)
print('TV L2: ', tv_image_l2)
print('TV L21: ', tv_image_l21)
print('TV Noise: ', tv_noise_image)
print('TV Noise L0: ', tv_noise_image_l0)
print('TV Noise L1: ', tv_noise_image_l1)
print('TV Noise L2: ', tv_noise_image_l2)
print('TV Noise L21: ', tv_noise_image_l21)
