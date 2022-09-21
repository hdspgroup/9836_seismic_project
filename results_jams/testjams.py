from Function import Sampling, Algorithms
import numpy as np
from scipy.io import savemat
from skimage.metrics import structural_similarity as ssim
from fktransform import fktransform
from matplotlib import pyplot as plt

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


def searchparameters(iter1, iter2, limits2=[0,5], iter3=1, limits3=[0,5], iter4=1, limits4=[0,5], algstr="twist", mstr = "aleatorio"):
    sampling = Sampling()
    seismic_data1 = load_seismic_data('/home/jams/Documents/9836_seismic_project/Desarrollo/ReDS/data/data.npy')

    jitparams = dict(gamma=3, epsilon=3)
    #mstr = "jitter"
    #algstr = "twist"
    step4 = (limits4[1]-limits4[0])/iter4
    step3 = (limits3[1] - limits3[0]) / iter3
    step2 = (limits2[1] - limits2[0]) / iter2
    psnrs = np.zeros((iter1, iter2, iter3, iter4))
    snrs = np.zeros((iter1, iter2, iter3, iter4))
    ssims = np.zeros((iter1, iter2, iter3, iter4))
    for r in range(iter4):
        p4 = limits4[0] + (step4*r)
        for j in range(iter3):
            p3 = limits3[0] + (step3 * j)
            for k in range(iter2):
                p2 = limits2[0] + (step2 * k)
                for i in range(iter1):
                    sampling_dict, H = sampling.apply_sampling(seismic_data1, mstr, jitparams, None, None, 0.33)
                    p_init = int(10 + np.floor(75 * np.random.rand()))
                    H[p_init:(p_init + 10)] = 0
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
                    if algstr == "twist":
                        params = dict(param1=p2, param2=p3, param3=p4)
                    elif algstr == "gap":
                        params = dict(param1=p2)
                    elif algstr == "fista":
                        params = dict(param1=p2, param2=p3)
                    iters = 150
                    algorithm, parameters = Alg.get_algorithm(algstr, iters, **params)

                    res = algorithm(**parameters)
                    for itr, res_dict in res:
                        if itr >= iters:
                            break
                    x_result, hist = next(res)
                    [x_result, _, _] = fktransform(x_result, 0.02, 10)
                    [seismic_data2, _, _] = fktransform(seismic_data1, 0.02, 10)
                    """plt.figure()
                    plt.imshow(x_result, cmap="seismic")
                    plt.show()
                    plt.figure()
                    plt.imshow(seismic_data2, cmap="seismic")
                    plt.show()"""
                    psnrs[i,k,j,r] = PSNR(seismic_data2, x_result)
                    snrs[i,k,j,r] = SNR(seismic_data2, x_result)
                    ssims[i,k,j,r] = ssim(seismic_data2, x_result)
                    print("iter: ", i, "iter2: ", k, "iter3: ", j, "iter4: ", r, "PSNR: ", psnrs[i,k,j,r], "SNR: ", snrs[i,k,j,r], " ssim: ", ssims[i,k,j,r])

                savemat('./FKparams-' + algstr + mstr + '33-r.mat', {'psnrs': psnrs, 'snrs': snrs, 'ssims': ssims})


def simulationsfixedparameters():
    sampling = Sampling()
    seismic_data1 = load_seismic_data('/home/jams/Documents/9836_seismic_project/Desarrollo/ReDS/data/data.npy')
    psnrs = np.zeros((30,1))
    snrs = np.zeros((30,1))
    ssims = np.zeros((30,1))
    jitparams = dict(gamma=3, epsilon=3)
    mstr = "aleatorio"
    algstr = "fista"
    for i in range(30):
        sampling_dict, H = sampling.apply_sampling(seismic_data1, mstr, jitparams, None, None, 0.33)
        #H = H==0
        #p_init = int(10 + np.floor(75*np.random.rand()))
        #H[p_init:(p_init+10)] = 0
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
        if algstr=="twist":
            params = dict(param1=2.25,param2=1.75,param3=3.0)
        elif algstr=="gap":
            params = dict(param1=10.8)
        elif algstr=="fista":
            params = dict(param1=2.4, param2=1.2)
        iters = 300
        algorithm, parameters = Alg.get_algorithm(algstr, iters, **params)

        res = algorithm(**parameters)
        for itr, res_dict in res:
            if itr>=iters:
                break
        x_result, hist = next(res)
        [x_result,_,_] = fktransform(x_result, 0.02, 10)
        [seismic_data2, _, _] = fktransform(seismic_data1, 0.02, 10)
        """plt.figure()
        plt.imshow(x_result, cmap="seismic")
        plt.show()
        plt.figure()
        plt.imshow(seismic_data2, cmap="seismic")
        plt.show()"""
        psnrs[i]=PSNR(seismic_data2,x_result)
        snrs[i] = SNR(seismic_data2, x_result)
        ssims[i]=ssim(seismic_data2,x_result)
        print("iter: ", i,"PSNR: ", psnrs[i],"SNR: ", snrs[i], " ssim: ", ssims[i])

    savemat('./FK-'+algstr+mstr+'33-s.mat',{'psnrs':psnrs,'snrs':snrs, 'ssims':ssims})

simulationsfixedparameters()
#searchparameters(3, 12, limits2=[0,3], iter3=12, limits3=[0,3], iter4=20, limits4=[0,5], algstr="twist", mstr = "aleatorio")
#searchparameters(3, 40, limits2=[0,15], iter3=1, limits3=[0,5], iter4=1, limits4=[0,5], algstr="gap", mstr = "aleatorio")
#searchparameters(3, 20, limits2=[0,3], iter3=20, limits3=[0,3], iter4=1, limits4=[0,5], algstr="fista", mstr = "aleatorio")