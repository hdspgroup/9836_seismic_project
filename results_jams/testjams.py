from Algorithms.Function import Sampling, Algorithms
import numpy as np
from scipy.io import savemat
from skimage.metrics import structural_similarity as ssim

def PSNR(original, compressed):
    '''
    The Peak Signal-to-Noise Ratio (PSNR) is an engineering term for
    the ratio between the maximum power of a signal and the power of the
    corrupting noise that affects the fidelity.

    This metrics is usually used to quantify the reconstruction quality
    for images and videos subject to a compression process.

    Mathematically, this term is defined as:

    .. math::
        PSNR = 10\log\left( \frac{MAX_I^2}{MSE} \right)

    Where:

    .. math::
        MSE = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1}[ I(i,j) - K(i,j) ]^2

    Parameters
    ----------
    original   : array-like
                 The original signal to compare.
    compressed : array-like
                 The reconstructed signal.

    Returns
    -------
    psnr : float
           The solution of the solf thresholding operator for the
           input array and threshold value.
    '''
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = np.max(original)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def SNR(original, compressed):
    '''
    The Peak Signal-to-Noise Ratio (PSNR) is an engineering term for
    the ratio between the maximum power of a signal and the power of the
    corrupting noise that affects the fidelity.

    This metrics is usually used to quantify the reconstruction quality
    for images and videos subject to a compression process.

    Mathematically, this term is defined as:

    .. math::
        PSNR = 10\log\left( \frac{MAX_I^2}{MSE} \right)

    Where:

    .. math::
        MSE = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1}[ I(i,j) - K(i,j) ]^2

    Parameters
    ----------
    original   : array-like
                 The original signal to compare.
    compressed : array-like
                 The reconstructed signal.

    Returns
    -------
    psnr : float
           The solution of the solf thresholding operator for the
           input array and threshold value.
    '''
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    power = np.mean(original ** 2)
    snr = 10 * np.log10(power / mse)
    return snr

def load_seismic_data(uploaded_directory):
    seismic_data = np.load(uploaded_directory)

    if seismic_data.ndim > 2:
        seismic_data = seismic_data[..., int(seismic_data.shape[-1] / 2)]
    else:  # only for data.npy
        seismic_data = seismic_data.T

    seismic_data = seismic_data / np.max(np.abs(seismic_data))

    return seismic_data


sampling = Sampling()
seismic_data1 = load_seismic_data('/home/jams/Documents/9836_seismic_project/data/data.npy')
psnrs = np.zeros((30,1))
snrs = np.zeros((30,1))
ssims = np.zeros((30,1))
jitparams = dict(gamma=3, epsilon=3)
mstr = "window"
algstr = "twist"
for i in range(30):
    sampling_dict, H = sampling.apply_sampling(seismic_data1, mstr, jitparams, None, None, 0.33)
    p_init = int(10 + np.floor(75*np.random.rand()))
    H[p_init:(p_init+10)] = 0
    Alg = Algorithms(seismic_data1, H, 'DCT2D', 'IDCT2D')  # Assuming using DCT2D ad IDCT2D for all algorithms
    '''
            if alg_name == 'FISTA':
                alg = self.FISTA
            elif alg_name == 'GAP':
                alg = self.GAP
            elif alg_name == 'TwIST':
                alg = self.TwIST
            elif alg_name == 'ADMM':
                alg = self.ADMM
    '''
    params = dict(param1=0.9,param2=1.89,param3=3.1)
    iters = 300
    algorithm, parameters = Alg.get_algorithm(algstr, iters, **params)

    res = algorithm(**parameters)
    for itr, res_dict in res:
        if itr>=iters:
            break
    x_result, hist = next(res)
    psnrs[i]=PSNR(seismic_data1,x_result)
    snrs[i] = SNR(seismic_data1, x_result)
    ssims[i]=ssim(seismic_data1,x_result)
    print("PSNR: ", psnrs[i],"SNR: ", snrs[i], " ssim: ", ssims[i])


savemat('./results/'+algstr+mstr+'33-r.mat',{'psnrs':psnrs,'snrs':snrs, 'ssims':ssims})