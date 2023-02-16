import matplotlib.pyplot as plt
from Algorithms.Function import *
import time
import matplotlib
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from tqdm import tqdm
import hdf5storage


def plot_results_traces(x, x_result, pattern_rand, case):
    # -------------- Visualization ----------------
    y_rand = x.copy()
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
    rem_traces = np.arange(len(pattern_rand))
    rem_traces = rem_traces[pattern_rand == 0]

    psnr_vec = []  # traces
    for s in rem_traces:
        psnr_vec.append(PSNR(x[:, s], x_result[:, s]))
    idxs = (-np.array(psnr_vec)).argsort()  # traces ordered

    axs[0, 0].imshow(x, cmap='seismic', aspect='auto')
    axs[0, 0].set_title(f'Reference')

    axs[1, 0].imshow(y_rand, cmap='seismic', aspect='auto')
    axs[1, 0].set_title(f'Incomplete')

    metric = PSNR(x, x_result)
    metric_ssim = ssim(x, x_result)
    axs[0, 1].imshow(x_result, cmap='seismic', aspect='auto')
    axs[0, 1].set_title(f'Reconstructed \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

    index = idxs[0]
    axs[1, 1].plot(x[:, rem_traces[index]], 'r', label='Reference')
    axs[1, 1].plot(x_result[:, rem_traces[index]], 'b', label='Recovered')
    axs[1, 1].legend(loc='best')
    plt.title('Trace ' + str("{:.0f}".format(rem_traces[index])))
    axs[1, 1].grid(axis='both', linestyle='--')
    axs[1, 1].set_title(f'Good Reconstruction')

    index = idxs[1]
    axs[0, 2].plot(x[:, rem_traces[index]], 'r', label='Reference')
    axs[0, 2].plot(x_result[:, rem_traces[index]], 'b', label='Recovered')
    axs[0, 2].legend(loc='best')
    plt.title('Trace ' + str("{:.0f}".format(rem_traces[index])))
    axs[0, 2].grid(axis='both', linestyle='--')
    axs[0, 2].set_title(f'Good Reconstruction')

    index = idxs[2]
    axs[1, 2].plot(x[:, rem_traces[index]], 'r', label='Reference')
    axs[1, 2].plot(x_result[:, rem_traces[index]], 'b', label='Recovered')
    axs[1, 2].legend(loc='best')
    plt.title('Trace ' + str("{:.0f}".format(rem_traces[index])))
    axs[1, 2].grid(axis='both', linestyle='--')
    axs[1, 2].set_title(f'Good Reconstruction')

    index = idxs[-1]
    axs[0, 3].plot(x[:, rem_traces[index]], 'r', label='Reference')
    axs[0, 3].plot(x_result[:, rem_traces[index]], 'b', label='Recovered')
    axs[0, 3].legend(loc='best')
    plt.title('Trace ' + str("{:.0f}".format(rem_traces[index])))
    axs[0, 3].grid(axis='both', linestyle='--')
    axs[0, 3].set_title(f'Bad Reconstruction')

    index = idxs[-2]
    axs[1, 3].plot(x[:, rem_traces[index]], 'r', label='Reference')
    axs[1, 3].plot(x_result[:, rem_traces[index]], 'b', label='Recovered')
    axs[1, 3].legend(loc='best')
    plt.title('Trace ' + str("{:.0f}".format(rem_traces[index])))
    axs[1, 3].grid(axis='both', linestyle='--')
    axs[1, 3].set_title(f'Bad Reconstruction')

    fig.tight_layout()
    plt.show()


def fastMarching_approach(data_path, data_format='numpy', exp_number=1, H=None):
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

        if len(x.shape) > 2:
            x = x[:, :, int(x.shape[-1] / 2)]
        '''
        ---------------  SAMPLING --------------------
        '''
        sr_rand = 0.5  # 1-compression
        _, _, H = random_sampling(x, sr_rand, seed=0)

        pattern_rand = [int(h) for h in H]
        pattern_rand = np.array(pattern_rand)
        # x -= x.min()
        # x /= x.max()
        y = x.copy()
        y[..., pattern_rand == 0] = 0

        mask = np.zeros(x.shape)
        mask[:, pattern_rand == 0] = 1
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

        imShape = x.shape
        x_copy = x.copy()
        x = x * (1 - mask)
        print(f"Running first reconstruction...")
        output = cv2.inpaint(x, mask, int(t_m/2) + 1, flags=paint_method)
        for i in tqdm(range(1, x.shape[-1] - 1)):
            # tmp = HaLRTC(x[:, :, [i - 1, i, i + 1]], y[:, :, [i - 1, i, i + 1]], mask[..., 0])
            tmp = cv2.inpaint(output[:, [i - 1, i, i + 1]], mask[:, [i - 1, i, i + 1]], t_m + 1, flags=paint_method)

            # output[..., i] = np.mean(tmp, axis=2)
            aux = output[:, [i - 1, i, i + 1]]
            aux = (tmp.astype('float32') + aux.astype('float32')) / 2

            output[:, [i - 1, i, i + 1]] = aux = aux.astype('uint8')  # cv2.medianBlur(aux, 3)
            # output[..., i] = np.mean(tmp, axis=2)
            # aux_s[[i - 1, i, i + 1]] += 1
        # output /= aux_s
        x = x_copy.copy()
        print(PSNR(x, output))
        r.append(PSNR(x, output))
        plot_results_traces(x, output, pattern_rand, 'Fast Marching (Inpainting)')

    print(f"Mean Result: {np.mean(r)}")


if __name__ == '__main__':
    data_name = 'data.npy'
    data_path = '../../Desarrollo/ReDS/data/' + data_name
    fastMarching_approach(data_path)
