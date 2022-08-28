import os.path

import matplotlib.pyplot as plt
from Algorithms.Function import random_sampling
from image_quality_metrics.utils.psnr import PSNR
from image_quality_metrics.utils.ssim import SSIM
from image_quality_metrics.utils.msssim import MSSSIM
from image_quality_metrics.utils.lpips import LPIPS
import time
import matplotlib
# from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from tqdm import tqdm
import hdf5storage
import pandas as pd


def compute_msssim(x, x_recon):
    x = np.tile(np.expand_dims(x, axis=2), [1, 1, 3])
    x_recon = np.tile(np.expand_dims(x_recon, axis=2), [1, 1, 3])
    return msssim(x, x_recon)


def compute_lpips(x, x_recon):
    x = np.tile(np.expand_dims(x, axis=2), [1, 1, 3])
    x_recon = np.tile(np.expand_dims(x_recon, axis=2), [1, 1, 3])
    return lpips(x, x_recon)


def save_metrics(x, x_result, pattern_rand, t_m_val):
    rem_shots = np.arange(len(pattern_rand))
    rem_shots = rem_shots[pattern_rand == 0]
    if not os.path.exists("params_results.xlsx"):

        column_names = ["Inpaint Radius", "Rem Shots", "PSNR Vec",
                        "SSIM Vec", "MSSSIM Vec", "LPIPS Vec",
                        "PSNR Mean", "SSIM Mean", "MSSSIM Mean", "LPIPS Mean"]

        df = pd.DataFrame(columns=column_names)
        df.to_excel("params_results.xlsx", index=False)
    else:
        df = pd.read_excel('params_results.xlsx')
    psnr_vec = []
    ssim_vec = []
    msssim_vec = []
    lpips_vec = []
    print("Saving results to params_results.xlsx")
    for rm_s in rem_shots:
        metric_psnr = psnr(x[..., rm_s], x_result[..., rm_s])
        metric_ssim = ssim(x[..., rm_s], x_result[..., rm_s])
        metric_msssim = compute_msssim(x[..., rm_s], x_result[..., rm_s])
        metric_lpips = compute_lpips(x[..., rm_s], x_result[..., rm_s])
        psnr_vec.append(metric_psnr)
        ssim_vec.append(metric_ssim)
        msssim_vec.append(metric_msssim)
        lpips_vec.append(metric_lpips)

    df2 = pd.DataFrame({
        "Inpaint Radius": str(t_m_val),
        "Rem Shots": str(rem_shots),
        "PSNR Vec": str(psnr_vec),
        "SSIM Vec": str(ssim_vec),
        "MSSSIM Vec": str(msssim_vec),
        "LPIPS Vec": str(lpips_vec),
        "PSNR Mean": str(np.mean(psnr_vec)),
        "SSIM Mean": str(np.mean(ssim_vec)),
        "MSSSIM Mean": str(np.mean(msssim_vec)),
        "LPIPS Mean": str(np.mean(lpips_vec))
    }, index=[0])

    df = pd.concat([df, df2], ignore_index=True, axis=0)
    df.to_excel("params_results.xlsx", index=False)


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
        plt.savefig(output_name + '_rm_shots.png')

    # x = Alg.x
    matplotlib.rcParams.update({'font.size': 8})
    fig, axs = plt.subplots(2, 4, dpi=250)
    fig.suptitle('Results from the ' + str(case) + ' Algorithm')
    rem_shots = np.arange(len(pattern_rand))
    rem_shots = rem_shots[pattern_rand == 0]
    psnr_vec = []
    for s in rem_shots:
        psnr_vec.append(psnr(x[..., s], x_result[..., s]))
    idxs = (-np.array(psnr_vec)).argsort()
    rem_shots = rem_shots[idxs]
    axs[0, 0].imshow(x[..., rem_shots[0]], cmap='seismic', aspect='auto')
    axs[0, 0].set_title(f'Reference, shot {rem_shots[0]}')

    axs[1, 0].imshow(x[..., rem_shots[1]], cmap='seismic', aspect='auto')
    axs[1, 0].set_title(f'Reference, shot {rem_shots[1]}')

    metric = psnr(x[..., rem_shots[0]], x_result[..., rem_shots[0]])
    metric_ssim = ssim(x[..., rem_shots[0]], x_result[..., rem_shots[0]])
    axs[0, 1].imshow(x_result[..., rem_shots[0]], cmap='seismic', aspect='auto')
    axs[0, 1].set_title(f'Reconstructed shot {rem_shots[0]}, \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

    metric = psnr(x[..., rem_shots[1]], x_result[..., rem_shots[1]])
    metric_ssim = ssim(x[..., rem_shots[1]], x_result[..., rem_shots[1]])
    axs[1, 1].imshow(x_result[..., rem_shots[1]], cmap='seismic', aspect='auto')
    axs[1, 1].set_title(f'Reconstructed shot {rem_shots[1]}, \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

    # ====
    axs[0, 2].imshow(x[..., rem_shots[2]], cmap='seismic', aspect='auto')
    axs[0, 2].set_title(f'Reference, shot {rem_shots[2]}')

    axs[1, 2].imshow(x[..., rem_shots[3]], cmap='seismic', aspect='auto')
    axs[1, 2].set_title(f'Reference, shot {rem_shots[3]}')

    metric = psnr(x[..., rem_shots[2]], x_result[..., rem_shots[2]])
    metric_ssim = ssim(x[..., rem_shots[2]], x_result[..., rem_shots[2]])
    axs[0, 3].imshow(x_result[..., rem_shots[2]], cmap='seismic', aspect='auto')
    axs[0, 3].set_title(f'Reconstructed shot {rem_shots[2]}, \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

    metric = psnr(x[..., rem_shots[3]], x_result[..., rem_shots[3]])
    metric_ssim = ssim(x[..., rem_shots[3]], x_result[..., rem_shots[3]])
    axs[1, 3].imshow(x_result[..., rem_shots[3]], cmap='seismic', aspect='auto')
    axs[1, 3].set_title(f'Reconstructed shot {rem_shots[3]}, \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

    fig.tight_layout()
    if len(output_name) == 0:
        plt.show()
    else:
        plt.savefig(output_name + '_recon.png')


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
        assert x.dtype == 'uint8'
        assert output.dtype == 'uint8'
        print(psnr(x, output))
        r.append(psnr(x, output))
        output_name = 'params_results/tm=' + str(t_m)
        np.savez(output_name + '.npz', x=x, output=output)
        plot_results(x.copy(), output.copy(), pattern_rand, f'Fast Marching (Inpainting), Inp. Radius: {t_m}',
                     output_name=output_name)
        save_metrics(x.copy(), output.copy(), pattern_rand, t_m)

    print(f"Mean Result: {np.mean(r)}")
    return np.mean(r)


if __name__ == '__main__':
    data_path = '/home/carlosh/Data_Seismic/RL3042.mat'
    dict_res = {}
    psnr = PSNR().forward
    ssim = SSIM().forward
    msssim = MSSSIM().forward
    lpips = LPIPS().forward
    for tm_val in range(0, 131, 3):
        print(f"Running experiment inpainting radius: {tm_val}")
        res = fastMarching_approach(data_path, 'matlab', tm_val)
        dict_res[tm_val] = res

    np.savez("params_results/all_res.npz", dict_res=dict_res)
