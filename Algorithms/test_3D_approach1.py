import os.path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
from Function import *
from skimage.metrics import structural_similarity as ssim
from mayavi import mlab

import scipy


def run_alg(data_name, case='ADMM', maxiter=500, sr_rand=0.5):
    # ----------------- --------------------
    # x = np.load('../data/data.npy')

    x = np.load('../data/' + data_name)

    if data_name == 'data.npy':
        x = x.T
    x = x / np.abs(x).max()

    '''
    ---------------  SAMPLING --------------------
    '''
    # sr_rand = 0.5  # 1-compression
    y_rand, pattern_rand, pattern_index = random_sampling(x[:, int(x.shape[1] / 2), :], sr_rand)
    H = pattern_index
    full_result = np.zeros(x.shape)
    '''
    ---------------- RECOVERY ALGORITHM -----------------
    Select the Algorithm: FISTA , GAP , TWIST , ADMM
    '''
    for s in range(x.shape[1]):
        Alg = Algorithms(x[:, s, :], H, 'DCT2D', 'IDCT2D')
        parameters = {}
        if case == 'FISTA':
            parameters = {'max_itr': maxiter,
                          'lmb': 0.1,
                          'mu': 0.3
                          }

        elif case == 'GAP':
            parameters = {'max_itr': maxiter,
                          'lmb': 1e-0
                          }

        elif case == 'TwIST':
            parameters = {'max_itr': maxiter,
                          'lmb': 0.5,
                          'alpha': 1.2,
                          'beta': 1.998
                          }

        elif case == 'ADMM':
            parameters = {'max_itr': maxiter,
                          'lmb': 5e-4,
                          'rho': 1,
                          'gamma': 1
                          }

        x_result, hist = Alg.get_results(case, **parameters)
        full_result[:, s, :] = x_result

    np.savez("arrays/" + data_name.split(".")[0] + "_Alg_" + case + "_maxIters_" + str(maxiter)
             + "_srRand_" + str(sr_rand) + ".npz",
             x=x, full_result=full_result, pattern_rand=pattern_rand, H=H, case=case, y_rand=y_rand)
    # --------------------------------


def plot_results(x, x_result, pattern_rand, case):
    # -------------- Visualization ----------------
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

    # 3D Visualization
    scalars = x[200:400, ...]
    fig = mlab.figure(figure='seismic', bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    mlab.volume_slice(scalars, slice_index=0, plane_orientation='x_axes', figure=fig)  # crossline slice
    mlab.volume_slice(scalars, slice_index=0, plane_orientation='y_axes', figure=fig)  # inline slice
    mlab.volume_slice(scalars, slice_index=0, plane_orientation='z_axes', figure=fig)  # depth slice
    mlab.axes(xlabel='Time', ylabel='Traces', zlabel='Shots', nb_labels=10)  # Add axes labels
    mlab.show()

    scalars = x_result[200:400, ...]
    fig = mlab.figure(figure='seismic', bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    mlab.volume_slice(scalars, slice_index=0, plane_orientation='x_axes', figure=fig)  # crossline slice
    mlab.volume_slice(scalars, slice_index=0, plane_orientation='y_axes', figure=fig)  # inline slice
    mlab.volume_slice(scalars, slice_index=0, plane_orientation='z_axes', figure=fig)  # depth slice
    mlab.axes(xlabel='Time', ylabel='Traces', zlabel='Shots', nb_labels=10)  # Add axes labels
    mlab.show()


if __name__ == '__main__':
    if not os.path.exists("arrays"):
        os.makedirs("arrays", True)
    data_n = 'cube4.npy'
    # run_alg(data_name=data_n, case='ADMM', maxiter=500, sr_rand=0.5)
    # run_alg(data_name=data_n, case='FISTA', maxiter=500, sr_rand=0.5)
    # run_alg(data_name=data_n, case='GAP', maxiter=500, sr_rand=0.5)
    # run_alg(data_name=data_n, case='TwIST', maxiter=500, sr_rand=0.5)
    #
    # data_n = 'spii15s.npy'
    # run_alg(data_name=data_n, case='ADMM', maxiter=500, sr_rand=0.5)
    # run_alg(data_name=data_n, case='FISTA', maxiter=500, sr_rand=0.5)
    # run_alg(data_name=data_n, case='GAP', maxiter=500, sr_rand=0.5)
    # run_alg(data_name=data_n, case='TwIST', maxiter=500, sr_rand=0.5)
    #
    # data_n = 'syn3D_cross-spread2.npy'
    # run_alg(data_name=data_n, case='ADMM', maxiter=500, sr_rand=0.5)
    # run_alg(data_name=data_n, case='FISTA', maxiter=500, sr_rand=0.5)
    # run_alg(data_name=data_n, case='GAP', maxiter=500, sr_rand=0.5)
    # run_alg(data_name=data_n, case='TwIST', maxiter=500, sr_rand=0.5)

    files = np.load("arrays/cube4_Alg_GAP_maxIters_500_srRand_0.5.npz")
    plot_results(files['x'], files['full_result'], files['pattern_rand'], files['case'])
