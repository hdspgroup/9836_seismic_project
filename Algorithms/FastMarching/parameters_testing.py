import matplotlib.pyplot as plt
from Algorithms.Function import *
import time
import matplotlib
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from tqdm import tqdm
import hdf5storage


def plot_results(x, x_result, pattern_rand, case, output_name=''):
    # -------------- Visualization ----------------
    y_rand = x[:, int(x.shape[1] / 2)].copy()
    y_rand[:, pattern_rand == 0] = 0

    plt.imshow(y_rand, cmap='seismic', aspect='auto')
    plt.title("Removed shots")
    plt.xlabel("Shots")
    plt.ylabel("Time")
    if len(output_name) == 0:
        plt.show()
    else:
        plt.savefig(output_name+'rm_shots.png')

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
    if len(output_name) == 0:
        plt.show()
    else:
        plt.savefig(output_name+'recon.png')


def fastMarching_approach(data_path, data_format='numpy', tm_val=0, exp_number=1, H=None):
    """

    Parameters
    ----------
    data_format: format of the data (Matlab or Numpy)
    data_path: path where the data is, the data should be in the format (time, traces, shots).
    exp_number: number of experiments. The results shows at the end is the average.
    H: Boolean vector indicating the removed shots

    Returns
    -------
    The reconstructed cube
    """
    r = []
    for exp in range(exp_number):
        if data_format == 'matlab':
            x = hdf5storage.loadmat(data_path)['RL3042']
        else:
            x = np.load(data_path)

        if 'data.npy' in data_path:
            x = x.T

        '''
        ---------------  SAMPLING --------------------
        '''
        if H is None:
            sr_rand = 0.5  # 1-compression
            _, _, H = random_sampling(x[:, int(x.shape[1] / 2), :], sr_rand)

        pattern_rand = [int(h) for h in H]
        pattern_rand = np.array(pattern_rand)
        # x -= x.min()
        # x /= x.max()
        y = x.copy()
        y[..., pattern_rand == 0] = 0
        x = np.transpose(x, [0, 2, 1])
        y = np.transpose(y, [0, 2, 1])
        mask = np.zeros(x.shape)
        mask[:, pattern_rand == 0, :] = 1
        x -= x.min()
        x /= x.max()
        x *= 255
        x = x.astype('uint8')
        mask = mask.astype('uint8')
        paint_method = cv2.INPAINT_TELEA
        output = np.zeros(x.shape)
        tmp = str(pattern_rand.astype('uint8')).split('1')
        t_m = 0
        for t in tmp:
            if t.strip().count('0') > t_m:
                t_m = t.strip().count('0')
        t_m = tm_val
        aux_s = np.zeros(x.shape[-1])
        imShape = x.shape
        x = np.reshape(x, [imShape[0] * imShape[1], imShape[2]])
        mask = np.reshape(mask, [imShape[0] * imShape[1], imShape[2]])
        x_copy = x.copy()
        x = x * (1 - mask)
        print(f"Running first reconstruction...")
        output = cv2.inpaint(x, mask, t_m + 1, flags=paint_method)
        output = np.reshape(output, [imShape[0], imShape[1], imShape[2]])
        x = np.reshape(x, [imShape[0], imShape[1], imShape[2]])
        mask = np.reshape(mask, [imShape[0], imShape[1], imShape[2]])
        for i in tqdm(range(1, x.shape[-1] - 1)):
            # tmp = HaLRTC(x[:, :, [i - 1, i, i + 1]], y[:, :, [i - 1, i, i + 1]], mask[..., 0])
            tmp = cv2.inpaint(x[:, :, [i - 1, i, i + 1]], mask[..., 0], t_m + 1, flags=paint_method)

            # output[..., i] = np.mean(tmp, axis=2)
            aux = output[..., [i - 1, i, i + 1]]
            aux = (tmp.astype('float32') + aux.astype('float32')) / 2

            output[..., [i - 1, i, i + 1]] = aux = aux.astype('uint8')  # cv2.medianBlur(aux, 3)
            # output[..., i] = np.mean(tmp, axis=2)
            # aux_s[[i - 1, i, i + 1]] += 1
        # output /= aux_s
        x_copy = np.reshape(x_copy, [imShape[0], imShape[1], imShape[2]])
        x = x_copy.copy()
        output = np.transpose(output, [0, 2, 1])
        x = np.transpose(x, [0, 2, 1])
        print(PSNR(x, output))
        r.append(PSNR(x, output))
        output_name = 'params_results/tm=' + str(t_m)
        np.savez(output_name+'.npz', output=output)
        plot_results(x, output, pattern_rand, 'Fast Marching (Inpainting)', output_name=output_name)

    print(f"Mean Result: {np.mean(r)}")
    return np.mean(r)


if __name__ == '__main__':
    data_path = '/home/carlosh/Data_Seismic/RL3042.mat'
    dict_res = {}
    for tm_val in range(0,101,2):
        res=fastMarching_approach(data_path, 'matlab')
        dict_res[tm_val] = res

    np.savez("params_results/all_res.npz", dict_res=dict_res)
