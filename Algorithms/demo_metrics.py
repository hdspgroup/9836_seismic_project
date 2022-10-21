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


# create a function that adds white gaussian noise to an image based on the SNR
def add_noise(image, snr):
    # calculate the signal power
    sig_power = np.sum(image ** 2) / image.size

    # calculate the noise power based on the SNR
    noise_power = sig_power / (10 ** (snr / 10))

    # generate a matrix of random values with the same dimensions as the image
    noise = np.random.normal(0, np.sqrt(noise_power), image.shape)

    # add the noise to the image
    noisy_image = image + noise

    return noisy_image

# load dataset sample

x = np.load('../Desarrollo/ReDS/data/shots/data_sample.npy')
x_noise = add_noise(x, 10)

xd = PSNR(x, x_noise)

