import numpy as np
from numpy.linalg import inv as inv
import numpy.linalg as ng
import imageio
import matplotlib.pyplot as plt
from Function import *
import time
import cv2
import matplotlib
from skimage.metrics import structural_similarity as ssim
import cv2
import tensorly as tl
import numpy as np
from numba import jit
from line_profiler import LineProfiler
from tqdm import tqdm


def shrinkage(X, t):
    U, Sig, VT = np.linalg.svd(X,full_matrices=False)
    Temp = np.zeros((U.shape[1], VT.shape[0]))
    for i in range(len(Sig)):
        Temp[i, i] = Sig[i]
    Sig = Temp

    Sigt = Sig
    imSize = Sigt.shape

    for i in range(imSize[0]):
        Sigt[i, i] = np.max(Sigt[i, i] - t, 0)

    temp = np.dot(U, Sigt)
    T = np.dot(temp, VT)
    return T


# @jit(nopython=True)
def ReplaceInd(X, known, Image):
    # imSize = Image.shape

    X[known[0], known[1], :] = Image[known[0], known[1], :]
    # for i in range(len(known)):
    #     in1 = int(np.ceil(known[i] / imSize[1]) - 1)
    #     in2 = int(imSize[0] - known[i] % imSize[1] - 1)
    #     X[in1, in2, :] = Image[in1, in2, :]
    return X


def HaLRTC(Image, X, mask):
    res = []
    known = np.where(mask == 0)
    imSize = Image.shape
    # test = np.zeros(Image.shape)
    # test = ReplaceInd(test, known, Image)
    a = abs(np.random.rand(3, 1))
    a = a / np.sum(a)
    p = 1e-6
    K = 50
    ArrSize = np.array(imSize)
    ArrSize = np.append(ArrSize, 3)
    Mi = np.zeros(ArrSize)
    Yi = np.zeros(ArrSize)

    for k in range(K):
        # compute Mi tensors(Step1)
        for i in range(ArrSize[3]):
            temp1 = shrinkage(tl.unfold(X, mode=i) + tl.unfold(np.squeeze(Yi[:, :, :, i]), mode=i) / p, a[i] / p)
            temp = tl.fold(temp1, i, imSize)
            Mi[:, :, :, i] = temp
        # Update X(Step2)
        X = np.sum(Mi - Yi / p, ArrSize[3]) / ArrSize[3]
        X = ReplaceInd(X, known, Image)
        # Update Yi tensors (Step 3)
        for i in range(ArrSize[3]):
            Yi[:, :, :, i] = np.squeeze(Yi[:, :, :, i]) - p * (np.squeeze(Mi[:, :, :, i]) - X)
        # Modify rho to help convergence(Step 4)
        p = 1.2 * p
    return X

#
# def fuc():
#     Image, X, known, a, Mi, Yi, imSize, ArrSize, p, K = init()
#
#     return X


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


if __name__ == '__main__':
    theta = 2
    alpha = 1000
    rho = 0.01
    beta = 0.1 * rho
    maxiter = 1000
    data_name = 'syn3D_cross-spread2.npy'

    r = []
    for exp in range(5):
        x = np.load('../data/' + data_name)

        if data_name == 'data.npy':
            x = x.T
        # x = x / np.abs(x).max()

        '''
        ---------------  SAMPLING --------------------
        '''
        sr_rand = 0.5  # 1-compression
        y_rand, pattern_rand, pattern_index = random_sampling(x[:, int(x.shape[1] / 2), :], sr_rand)
        H = pattern_index
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
        aux_s = np.zeros(x.shape[-1])
        imShape = x.shape
        x = np.reshape(x, [imShape[0] * imShape[1], imShape[2]])
        mask = np.reshape(mask, [imShape[0] * imShape[1], imShape[2]])
        output = cv2.inpaint(x, mask, t_m + 1, flags=paint_method)
        output = np.reshape(output, [imShape[0], imShape[1], imShape[2]])
        x = np.reshape(x, [imShape[0], imShape[1], imShape[2]])
        mask = np.reshape(mask, [imShape[0], imShape[1], imShape[2]])
        for i in tqdm(range(1, x.shape[-1] - 1)):
            # tmp = HaLRTC(x[:, :, [i - 1, i, i + 1]], y[:, :, [i - 1, i, i + 1]], mask[..., 0])
            tmp = cv2.inpaint(x[:, :, [i - 1, i, i + 1]], mask[..., 0], t_m + 1, flags=paint_method)

            # output[..., i] = np.mean(tmp, axis=2)
            aux = output[..., [i - 1, i, i + 1]]
            aux = (tmp.astype('float32') + aux.astype('float32'))/2

            output[..., [i - 1, i, i + 1]] = aux = aux.astype('uint8') # cv2.medianBlur(aux, 3)
            # output[..., i] = np.mean(tmp, axis=2)
            # aux_s[[i - 1, i, i + 1]] += 1
        # output /= aux_s
        output = np.transpose(output, [0, 2, 1])
        x = np.transpose(x, [0, 2, 1])
        print(PSNR(x, output))
        r.append(PSNR(x, output))
        plot_results(x, output, pattern_rand, 'Fast Marching (Inpainting)')
        # x = x[:, :3, :]
        # y = y[:, :3, :]
        # y = np.reshape(y, [x.shape[0] * x.shape[1], x.shape[2]])
        # x = np.reshape(x, [x.shape[0] * x.shape[1], x.shape[2]])
        # y = np.transpose(y, [0, 2, 1])
        # x = np.transpose(x, [0, 2, 1])
        # x = np.repeat(x[:, :, np.newaxis], 3, axis=2)
        # y = np.repeat(y[:, :, np.newaxis], 3, axis=2)
        '''
        print("Starting Algorithm")
        start = time.time()
        image_hat = GLTC_Geman(x, y, alpha, beta, rho, theta, maxiter)
    
        # image_hat = GLTC(x, y, alpha, beta, rho, maxiter)
        end = time.time()
        print(f"Duration: {end - start}")
        image_rec = np.round(image_hat).astype(int)
        image_rec[np.where(image_rec > 255)] = 255
        image_rec[np.where(image_rec < 0)] = 0
        pos = np.where((x != 0) & (y == 0))
        rse = np.linalg.norm(image_rec[pos] - x[pos], 2) / np.linalg.norm(x[pos], 2)
        np.savez("tmp.npz", image_rec=image_rec, rse=rse)
        '''

    print(f"Mean Result: {np.mean(r)}")
