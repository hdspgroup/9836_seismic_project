import numpy as np
import matplotlib.pyplot as plt


def fk(data, dt, dx):
    '''
    Compute the two-dimensional Discrete Fourier Transform.

    Parameters
    ----------
    data : array-like
           seismic data
    dt   : integer
           time interval
    dx   : integer
           trace interval

    Returns
    -------
    FK   : ndarray
           The transformed input along the [0, 1] axis
    f    : 1D array
    kx   : 1D array
    '''
    nt = data.shape[0]
    nx = data.shape[1]
    nt_fft = 2 * nt
    nx_fft = 2 * nx
    data_f = np.fft.fft(data, n=nt_fft, axis=0)
    data_fk = np.fft.fft(data_f, n=nx_fft, axis=1)
    FK = 20 * np.log10(np.fft.fftshift(np.abs(data_fk)))
    FK = FK[nt:, :]
    f = np.linspace(-0.5, 0.5, nt_fft) / dt
    kx = np.linspace(-0.5, 0.5, nt_fft) / dx
    return FK, f, kx


def FK_visualize(data, kx, samples):
    '''
    Parameters
    ----------
    data    : 2D array-like
              seismic data
    kx      : ndarray
    samples : integer
              number of time samples to visualize

    Returns
    -------
    Plot for visualizing any shot in frequency-wavenumber domain
    '''
    plt.imshow(data[:samples], aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
    plt.xlabel('Wavenumber 1/m')
    plt.ylabel('Frequency (Hz)')
    # plt.colorbar()
