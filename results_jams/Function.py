import scipy.io
import numpy as np
from skimage import transform
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
import inspect
try:
    from pyct.fdct2 import fdct2
except ImportError:
    from pyct.fdct2 import fdct2
    print("segundo intento")
import scipy.sparse.linalg as ln
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import time
# from pymitter import EventEmitter

# Need to use this (EventEmitter) for comunication with the GUI, please don't remove it, I used this trough the code
from Algorithms.tv_norm import tv_norm
#from Desarrollo.ReDS.gui.scripts.alerts import showWarning, showCritical


class Sampling:
    '''
    Sampling techniques for seismic data.
    '''

    def apply_sampling(self, x, mode, jitter_params, lista, seed, compression_ratio):
        if mode == 'aleatorio':
            return self.random_sampling(x, seed, compression_ratio)
        elif mode == 'regular':
            return self.uniform_sampling(x, compression_ratio)
        elif mode == 'jitter':
            # if jitter_params['gamma'] % 2 != 0:
            #     showWarning("El valor de gamma debe ser un número impar.")
            #     return

            return self.jitter_sampling(x, seed, **jitter_params)
        else:  # mode == lista
            try:
                lista = [int(number) for number in lista.text().replace(' ', '').split(',')]


            except:
                pass
                return
            return self.list_sampling(x, lista)

    def random_sampling(self, x, seed, compression_ratio):
        '''
        Inputs:
        x : full data
        seed: seed for random sampling
        compression_ratio: compression ratio for subsampling

        Outputs:
        sampling_dict : dictionary that contains all information about sample
                        and its compression
        '''
        M, N = x.shape

        # sampling
        tasa_compression = int(compression_ratio * N)
        pattern_vec = np.ones((N,))

        if seed is not None:
            np.random.seed(seed)

        ss = np.random.permutation(list(range(1, N - 1)))
        pattern_vec[ss[0:tasa_compression]] = 0
        H0 = np.tile(pattern_vec.reshape(1, -1), (M, 1))

        out = x * H0
        pattern_bool = np.asarray(pattern_vec, dtype=bool)
        H = pattern_bool

        sampling_dict = {
            "x_ori": x,
            "sr_rand": compression_ratio,
            "y_rand": out,
            "pattern_rand": pattern_vec,
            "pattern_index": pattern_bool,
            "H": H,
            "H0": H0
        }

        return np.array(list(sampling_dict.items())), H

    def uniform_sampling(self, x, compression_ratio):
        '''
        Inputs:
        x : full data
        compression_ratio: compression ratio for subsampling

        Outputs:
        sampling_dict : dictionary that contains all information about sample
                        and its compression
        '''
        M, N = x.shape
        pattern_vec = np.ones((N,), dtype=int)
        n_col_rmv = np.round(N * compression_ratio)
        x_distance = np.round(N / n_col_rmv)

        i = 0
        while i * int(x_distance) < N:
            pattern_vec[i * int(x_distance)] = 0
            i = i + 1

        # Sampling pattern
        H0 = np.tile(pattern_vec.reshape(1, -1), (M, 1))

        out = x * H0
        pattern_bool = np.asarray(pattern_vec, dtype=bool)
        H = pattern_bool

        sampling_dict = {
            "x_ori": x,
            "sr_rand": compression_ratio,
            "y_rand": out,
            "pattern_rand": pattern_vec,
            "pattern_index": pattern_bool,
            "H": H,
            "H0": H0
        }

        return np.array(list(sampling_dict.items())), H


    def jitter_sampling(self, x, seed, gamma=3, epsilon=3):
        # https://slim.gatech.edu/Publications/Public/Journals/Geophysics/2008/hennenfent08GEOsdw/paper_html/node14.html
        '''
        Inputs:
        x : full data
        gamma: measurements distance, which implies that the compression ratio is N/gamma
        epsilon: window where the perturbation is selected
        new_position_i = ((1-gamma)/2)*gamma*i + U(-epsilon/2, epsilon/2)

        Outputs:
        sampling_dict : dictionary that contains all information about sample
                        and its compression
        '''
        M, N = x.shape

        # sensing pattern (zero when not measure that position)
        pattern_vec = np.ones((N,))
        # first pisition
        init_value = ((1 - gamma) / 2) + gamma
        # compute the centroids of the regular grid sampling
        centroids = list(range(int(init_value) - 1, N - 1, gamma))
        # compute the limits of the uniform random variable
        limits = np.floor(epsilon / 2)
        # add 0.49 in order to make all number equally probable (edges problem)
        limits = limits + 0.49

        if seed is not None:
            np.random.seed(seed)

        # generate the perturbation following U(-epsilon/2, epsilon/2)
        res = (np.random.rand(len(centroids), ) * (2 * limits)) - limits
        # convert to integer
        res[res > 0] = np.floor(res[res > 0])
        res[res < 0] = np.ceil(res[res < 0])
        # apply the perturbation to the centroids
        positions = centroids - res
        # placing a one in the new position
        pattern_vec[positions.astype(int)] = 0

        # Sampling pattern
        H0 = np.tile(pattern_vec.reshape(1, -1), (M, 1))
        out = x * H0
        pattern_bool = np.asarray(pattern_vec, dtype=bool)
        H = pattern_bool

        sampling_dict = {
            "x_ori": x,
            "sr_rand": 1 / gamma,
            "y_rand": out,
            "pattern_rand": pattern_vec,
            "pattern_index": pattern_bool,
            "H": H,
            "H0": H0
        }
        return np.array(list(sampling_dict.items())), H

    def list_sampling(self, x, lista):
        M, N = x.shape

        # sampling

        pattern_vec = np.ones((N,))
        pattern_vec[np.array(lista)] = 0
        H0 = np.tile(pattern_vec.reshape(1, -1), (M, 1))

        out = x * H0
        pattern_bool = np.asarray(pattern_vec, dtype=bool)
        H = pattern_bool

        sampling_dict = {
            "x_ori": x,
            "lista": lista,
            "y_rand": out,
            "pattern_rand": pattern_vec,
            "pattern_index": pattern_bool,
            "H": H,
            "H0": H0
        }

        return np.array(list(sampling_dict.items())), H


def random_sampling(x, sr, seed=None):
    '''
    Random sampling is a part of the sampling technique in which each sample has an equal probability of being chosen.
    A sample chosen randomly is meant to be an unbiased representation of the total population.

    Attributes
    ----------
    x : array-like
        full data to apply the random sampling method
    sr: float
        subsampling factor
    '''
    dim = x.shape
    batch = 1  # dim[0]
    M = dim[0]
    N = dim[1]
    # L = dim[3]
    # sampling
    tasa_compression = int(sr * N)
    pattern_vec = np.ones((N,))

    if seed is not None:
        np.random.seed(seed)

    ss = np.random.permutation(list(range(1, N - 1)))
    pattern_vec[ss[0:tasa_compression]] = 0
    H0 = np.tile(pattern_vec.reshape(1, -1), (M, 1))

    out = x * H0

    pattern_bool = np.asarray(pattern_vec, dtype=bool)

    return out, pattern_vec, pattern_bool


def dct2():
    '''
    Returns a function to compute the Discrete Cosine
    Transform (DCT) for 2D signals.

    The DCT is a transform similar to the Discrete Fourier
    Transform, but using only real numbers. The DCT express
    a finite sequence of points in terms of cosine functions
    oscillating at different frequencies.

    The 1D DCT is computed as:

    .. math::
        y_{k} = 2\sum_{n=0}^{N-1} x_{n}\cos\left( \frac{\pi k (2n + 1)}{2N} \right)

    To compute the 2D transform, the 1D transform is applied
    to the rows and the columns of the input matrix.
    '''

    def dct2_function(x):
        return (scipy.fft.dct(scipy.fft.dct(x).T)).T

    return dct2_function


def idct2():
    '''
    Returns a function to compute the Inverse of the Discrete
    Cosine Transform (IDCT) for 2D signals.

    Formally, the Discrete Fourier Transform in a linear function
    that maps a real vector to another real vector of the same
    dimension. As the transform is a linear function, then
    invertible, it is possible to define an inverse function that
    allows to recover the original signal from the transformed
    values.

    The 1D IDCT is computed as:

    .. math::
        x_{k} = \frac{y_0}{2N} + \frac{1}{N}\sum_{n=1}^{N-1}y_{n}\cos\left( \frac{\pi (2k+1) n}{2N} \right)

    To compute the 2D transform, the 1D transform is applied
    to the rows and the columns of the input matrix.
    '''

    def idct2_function(x):
        return (scipy.fft.idct(scipy.fft.idct(x).T)).T

    return idct2_function


class Operator:
    '''
    A class to represent the matrix operators used in the
    forward model of the seismic reconstruction problem.

    Attributes
    ----------
    H : array-like
        The sensing matrix. A matrix with the positions of the
        missing elements.
    m : int
        The first dimension of the input data in a 2D form.
    n : int
        The second dimension of the input data in a 2D form.
    operator_dir : function
        This function applies a 2D transform to promote the
        sparsity of the signal in certain base of representation.
    operator_inv : function
        This function applies the inverse transform of the
        `operator_dir` function.

    Methods
    -------
    transpose(x)
        This method multiplies the input vector with the transpose
        of the operator.
    direct(x)
        This method multiplies the input vector with the equivalent
        of the operator for the model.
    '''

    def __init__(self, H, m, n, operator_dir, operator_inv, operator):#jams: added operator since the the behavior is different for curvelet
        '''
        Parameters
        ----------
        H : array-like
            The sensing matrix. A matrix with the positions of the
            missing elements.
        m : int
            The first dimension of the input data in a 2D form.
        n : int
            The second dimension of the input data in a 2D form.
        operator_dir : function
            This function applies a 2D transform to promote the
            sparsity of the signal in certain base of representation.
        operator_inv : function
            This function applies the inverse transform of the
            `operator_dir` function.
        '''
        self.H = H
        self.m = m
        self.n = n
        self.operator_dir = operator_dir
        self.operator_inv = operator_inv
        self.operator = operator

    def transpose(self, x):  # y = D'H' * x
        '''
        Applies the equivalent of the matricial transpose
        operator to the input vector.

        Mathematically is defined as

        .. math::
            y = D^T H^T x

        where H is the sensing matrix and D the transformation
        basis.

        Parameters
        ----------
        x : array-like
            An array to apply the operator.

        Returns
        -------
        y : array-like
            The traspose operation applied to the input
            vector.
        '''
        if self.operator == 'DCT2D':
            Ht = self.H.transpose()
            y = Ht * np.squeeze(x.T.reshape(-1))  # H' * x

            y = np.reshape(y, [self.m, self.n], order='F')

            y = self.operator_dir(y)
        else:
            Ht = self.H.transpose()
            y = Ht * np.squeeze(x)  # H' * x

            y = np.reshape(y, [self.n, self.m]) #data

            y = self.operator_dir(y) # curvelet vector

        return y

    def direct(self, x):  # y = H * D * x
        '''
        Applies the equivalent of the matricial direct
        operator to the input vector.

        Mathematically is defined as

        .. math::
            y = H D x

        where H is the sensing matrix and D the transformation
        basis.

        Parameters
        ----------
        x : array-like
            An array to apply the operator.

        Returns
        -------
        y : array-like
            The traspose operation applied to the input
            vector.
        '''
        #x = np.reshape(x, [self.m, self.n], order='F')  # ordenar
        if self.operator == 'DCT2D':
            x = np.reshape(x, [self.m, self.n], order='F')  # ordenar

            theta = self.operator_inv(x)  # D * x

            y = self.H * np.squeeze(theta.T.reshape(-1))  # H * D * x
        else:
            theta = self.operator_inv(x)  # D * x

            y = self.H * np.squeeze(theta.reshape(-1))  # H * D * x

        return y


# -------------------------------------------------------------------------
def soft_threshold(x, t):
    '''
    The Soft thresholding is a wavelet shrinkage operator.

    This operator is used in the area of compressive as the close
    solution of the proximal operator for the L1 norm.

    Parameters
    ----------
    x : array-like
        An array to apply the shrinkage operator.
    t : float
        The threshold value to compute the operator.

    Returns
    -------
    y : array-like
        The solution of the solf thresholding operator for the
        input array and threshold value.
    '''
    tmp = (np.abs(x) - t)
    tmp = (tmp + np.abs(tmp)) / 2
    y = np.sign(x) * tmp
    return y


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


class Algorithms:
    '''
    A class that contains different algorithms solutions to solve the
    seismic data reconstruction problem.

    Mathematically, the reconstruction problem is defined as an
    optimization problem with the form:


    where H is the sensing matrix, D is a transformation basis and
    x are the relative coefficients in the transformed domain.

    Attributes
    ----------
    x : array-like
        An 2D input image. The array is resized to the closest
        dimensions with the form 2^n.
    m : int
        The first dimension of the input data x in a 2D form.
    n : int
        The second dimension of the input data in a 2D form.
    H : array-like
        The sensing matrix. A matrix with the positions of the
        missing elements.
    operator_dir : function
        This function applies a 2D transform to promote the
        sparsity of the signal in certain base of representation.
        It could be a function or a string with the name of a
        predefined transform. Actually the only predefined transform
        is DCT and is used with the string 'DCT2D'.
    operator_inv : function
        This function applies the inverse transform of the
        `operator_dir` function.
        It could be a function or a string with the name of a
        predefined transform. Actually the only predefined transform
        is DCT and is used with the string 'IDCT2D'.

    Methods
    -------
    FISTA(lmb, mu, max_itr)
        Applies a Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
        to solve the optimization problem.
    GAP(lmb, max_itr)
        Applies a GAP Algorithm to solve the optimization problem.
    TwIST(lmb, alpha, beta, max_itr)
        Applies a Time to Walking Independently After Stroke (TwIST)
        algorithm to solve the optimization problem.
    '''

    def __init__(self, x, H, operator_dir, operator_inv):
        '''
        Parameters
        ----------
        x : array-like
            An 2D input image.
        H : array-like
            The sensing matrix. A matrix with the positions of the
            missing elements.
        operator_dir : function
            This function applies a 2D transform to promote the
            sparsity of the signal in certain base of representation.
        operator_inv : function
            This function applies the inverse transform of the
            `operator_dir` function.
        '''
        self.is_complete_data = True
        # ------ Build sensing matrix for incomplete data ---------
        if H is None:
            x = np.nan_to_num(x, nan=0)
            H = np.all(x != 0, axis=0)
            self.is_complete_data = False

        # ------- change the dimension of the inputs image --------
        m, n = x.shape
        # m = int(2 ** (np.ceil(np.log2(m)) - 1))
        # n = int(2 ** (np.ceil(np.log2(n)) - 1))
        x = transform.resize(x, (m, n))
        x = x / np.abs(x).max()
        self.x = x
        self.m, self.n = x.shape
        self.pattern = 0
        self.H_raw = H

        # ---------- Load or build the sensing matrix -------------
        if isinstance(H, (list, tuple, np.ndarray)):
            if (len(H.shape) == 1):
                # Create H from a given input pattern
                # The input H must be the pattern, i.e., H is a vector!
                temp0 = np.reshape(np.asarray(range(0, m * n)), [n, m]).T
                temp = temp0[:, H]
                self.pattern = H
                temp = temp.T.reshape(-1)  # Column Vectorization
                self.H = csr_matrix((np.ones(temp.shape), (range(0, len(temp)), temp)), shape=(len(temp), int(m * n)))
            else:
                # Takes the input H as the sampling matrix
                self.H = H
        else:
            # H here is a given subsampling value
            # Load a pre-determinated random pattern
            Nsub = int(np.round(m * (H)))
            # iava = np.random.permutation(m)
            iava = np.squeeze(loadmat('data/iava.mat')['iava']) - 1
            self.cort = np.sort(iava[Nsub:])
            iava = np.sort(iava[0:Nsub])
            self.pattern = iava
            temp0 = np.reshape(np.asarray(range(0, m * n)), [n, m]).T
            temp = temp0[:, iava]
            temp = temp.T.reshape(-1)  # Column Vectorization
            self.H = csr_matrix((np.ones(temp.shape), (range(0, len(temp)), temp)), shape=(len(temp), int(m * n)))

        # ---------- Load or create the basis function  ---------

        if inspect.isfunction(operator_dir):
            self.operator_dir = operator_dir
        else:
            if operator_dir == 'DCT2D':
                self.operator_dir = dct2()
            elif operator_dir == 'FDCT2':
                self.objtransform = fdct2((n, m), 4, 16, ac=True, norm=False)
                self.operator_dir = self.objtransform.fwd

        if inspect.isfunction(operator_inv):
            self.operator_inv = operator_inv
        else:
            if operator_inv == 'IDCT2D':
                self.operator_inv = idct2()
            elif operator_dir == 'FDCT2':
                self.operator_inv = self.objtransform.inv

        # ------------ This is a special class of operator ------------
        self.A = Operator(self.H, self.m, self.n, self.operator_dir, self.operator_inv, operator_dir)

        # ------------ Deleted traces vector --------------------------
        H_elim = np.linspace(0, len(self.pattern) - 1, len(self.pattern), dtype=int)
        self.H_elim = H_elim[np.invert(self.pattern)]

    def measurements(self):
        '''
        Operator measurement models the subsampled acquisition process given a
        sampling matrix H

        Returns
        -------
        measures : Y = H@x
        '''

        print(f'H.shape={self.H.shape}')
        return self.H * np.squeeze(self.x.T.reshape(-1))

    def get_results(self, alg_name, **parameters):
        '''
        This function allows to get the final results of the implemented algorithms
        in this class.

        This is due to algorithms functions works as generators, where each iteration
        returns the current output info of the algorithm and the last iteration
        returns the desired output of the function.

        Parameters
        ----------
        alg_name :    str
                      The name of the algorithm to solve.
        max_itr :     int
                      The maximum number of iteration for the algorithm.
        parameters :  dict
                      Parameters of the selected algorithm to solve.

        Returns
        -------
        x_results : recovery results of the selected algorithm.
        hist      : history of the selected algorithm.
        '''
        if alg_name == 'FISTA':
            alg = self.FISTA
        elif alg_name == 'GAP':
            alg = self.GAP
        elif alg_name == 'TwIST':
            alg = self.TwIST
        elif alg_name == 'ADMM':
            alg = self.ADMM
        else:
            raise 'The algorithm entered was not found.'

        results = [output for i, output in enumerate(alg(**parameters)) if parameters["max_itr"] == i][0]
        x_result, hist = results

        return x_result, hist

    def get_algorithm(self, algorithm_case, maxiter, **params):
        if algorithm_case == "fista":
            parameters = {"lmb": float(params['param1']),  # Tau
                          "mu": float(params['param2']),  # Mu
                          "max_itr": maxiter}
            func = self.FISTA

        elif algorithm_case == "gap":
            parameters = {"lmb": float(params['param1']),  # Tau
                          "max_itr": maxiter}
            func = self.GAP

        elif algorithm_case == "twist":
            parameters = {"lmb": float(params['param1']),  # Tau
                          "alpha": float(params['param2']),  # Alpha
                          "beta": float(params['param3']),  # Beta
                          "max_itr": maxiter}
            func = self.TwIST

        elif algorithm_case == "admm":
            parameters = {"rho": float(params['param1']),  # Rho
                          "gamma": float(params['param2']),  # Gamma
                          "lmb": float(params['param3']),  # Lambda
                          "max_itr": maxiter}
            func = self.ADMM

        else:
            return

        return func, parameters

    # ---------------------------------------------FISTA----------------------------------------

    def FISTA(self, lmb, mu, max_itr):
        '''
        This is the python implementation of the FISTA (A Fast Iterative Shrinkage-Thresholding Algorithm)
        Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear
        inverse problems. SIAM journal on imaging sciences, 2(1), 183-202.
        This one of the most well-known first-order optimization scheme in the literature, as it achieves
        the worst-case \mathbf{\mathit{O}}(1/k^{2}) optimal convergence rate in terms of objective function value.

        The FISTA is computed as:

        .. math::
              t_{k}=\frac{1+\sqrt{4t^{2}_{k-1}}}{2}
              a_{k}=\frac{t_{k-1}-1}{t_{k}}
              y_{k}=x_{k}+a_{k}(x_{k}-x_{k-1})
              x_{k+1}=prox_{\gamma R}(y_{k}-\gamma \bigtriangledown F(y_{k}))

        Input:
            self:       They have the variables of the Algorithm class, such as H,y, sparsity basis.
            lmb:        float
                        The sparsity regularizer
            mu :        type
                        The step-descent of the algorithm
            max_itr :   int
                        The maximum number of iterations
        '''
        # ee = EventEmitter()

        y = self.measurements()

        # print('FISTA: \n')

        dim = self.x.shape
        x = self.A.transpose(y)#np.zeros(dim)
        q = 1
        s = x
        hist = np.zeros((max_itr + 1, 4))
        # print('itr \t ||x-xold|| \t PSNR \n')
        itr = 0
        while (itr < max_itr):
            x_old = x
            s_old = s
            q_old = q
            temp = self.A.direct(s_old) - y
            grad = self.A.transpose(temp)  # Ht * (H * s_old - y)
            z = s_old - mu * (grad)

            # proximal
            x = soft_threshold(z, lmb)
            q = 0.5 * (1 + np.sqrt(1 + 4 * (q_old ** 2)))
            s = x + ((q_old - 1) / (q)) * (x - x_old)
            itr = itr + 1

            residualx = np.linalg.norm(x - x_old) / np.linalg.norm(x)

            # if self.is_complete_data:
            psnr_val = PSNR(self.x[:, :], np.transpose(self.operator_inv(s))[:, :])
            ssim_val = ssim(self.x[:, :], np.transpose(self.operator_inv(s))[:, :])
            tv_val = tv_norm(self.operator_inv(s))

            hist[itr, 0] = residualx
            hist[itr, 1] = psnr_val
            hist[itr, 2] = ssim_val
            hist[itr, 3] = tv_val
            nz_x = (x != 0.0) * 1
            num_nz_x = np.sum(nz_x)
            f = 0.5 * np.sum(temp * temp) + lmb * np.sum(np.abs(x))

            print(itr, '\t Error:', format(hist[itr, 0], ".2e"), '\t Obj:', format(f, ".2e"), '\t nz:',
                  format(num_nz_x, "d"), '\t PSNR:', format(hist[itr, 1], ".3f"), 'dB',
                  '\t SSIM:', format(hist[itr, 2], ".3f"), '\t TV norm: ', format(hist[itr, 3], ".2f"), '\n')

            yield itr, dict(result=self.operator_inv(s), hist=hist)

            # else:
            #     yield None, None

        yield self.operator_inv(s), hist

    # ---------------------------------------------GAP----------------------------------------
    def GAP(self, lmb, max_itr):
        '''
        Let \Phi \in \mathbb{R}^{r\times x}  with r < n be given and fixed, and z \in \mathbb{R}^{n}
        be an arbitrary S-sparse vector, which has at most S < r nonzero elements.
        Reconstruction of z from y = \Phi z is a problem that centers around the theory and practice
        of compressive sensing. The GAP algorithm extends classical alternating projection to the case in
        which projections are performed between convex sets that undergo a systematic sequence of changes.

        .. math::
                w_{t} = \theta_{t-1} + \Phi^{T}(\Phi\Phi^{T})(y - \Phi\theta_{t-1})

        The algorithm  can be interrupted anytime to return a valid solution and resumed subsequently to
        improve the  solution.

        https://arxiv.org/pdf/1511.03890.pdf

        Parameters
        ----------
        lmb :    float
                 The threshold value to compute the operator.
        max_itr : int
                  Maximum number of iterations
        '''
        y = self.measurements()

        #print('---------GAP method---------- \n')

        #dim = self.x.shape
        x = self.A.transpose(y)
        hist = np.zeros((max_itr + 1, 4))

        residualx = 1
        tol = 1e-2

        #print('itr \t ||x-xold|| \t PSNR \n')
        itr = 0
        while (itr < max_itr):  # & residualx>tol):
            x_old = x

            temp = self.A.direct(x) - y

            grad = self.A.transpose(temp)
            z = x - grad

            # proximal
            x = soft_threshold(z, lmb)
            itr = itr + 1

            residualx = np.linalg.norm(x - x_old) / np.linalg.norm(x)

            f = 0.5 * np.sum(temp * temp) + lmb * np.sum(np.abs(x))
            if self.A.operator == 'DCT2D':
                psnr_val = PSNR(self.x[:, :], (self.operator_inv(x))[:, :]) #
                ssim_val = ssim(self.x[:, :], (self.operator_inv(x))[:, :])
            else:
                psnr_val = PSNR(self.x[:, :], np.transpose(self.operator_inv(x))[:, :])  # np.transpose
                ssim_val = ssim(self.x[:, :], np.transpose(self.operator_inv(x))[:, :])
            tv_val = tv_norm(self.operator_inv(x))

            hist[itr, 0] = residualx
            hist[itr, 1] = psnr_val
            hist[itr, 2] = ssim_val
            hist[itr, 3] = tv_val
            nz_x = (x != 0.0) * 1
            num_nz_x = np.sum(nz_x)

            print(itr, '\t Error:', format(hist[itr, 0], ".2e"), '\t Obj:', format(f, ".2e"), '\t nz:', format(num_nz_x, "d"), '\t PSNR:', format(hist[itr, 1], ".3f"), 'dB',
                  '\t SSIM:', format(hist[itr, 2], ".3f"), '\t TV norm: ', format(hist[itr, 3], ".2f"), '\n')

            yield itr, dict(result=self.operator_inv(x), hist=hist)

        yield self.operator_inv(x), hist

    # ----------------TWIST----------------------------
    def TwIST(self, lmb, alpha, beta, max_itr):
        '''
        Stationary Two-Step Iterative Shrinkage/Thresholding (TWIST) for solving \mathbf{Ax=b}.
        Consider the linear system \mathbf{Ax=b}, with \mathbf{A} positive definite;
        define a so-called splitting of \mathbf{A} as \mathbf{A=C-R}, such that \mathbf{C}
        is positive definite and easy to invert. TWIST is defined as:

        .. math::
              \mathbf{x_{1}}=\mathbf{\Gamma_{\lambda}({x_{0})}}
              \mathbf{x_{t+1}}=(1-\alpha)\mathbf{x_{1}}+(\alpha-\beta)\mathbf{x_{t}}+\beta\mathbf{\Gamma_{\lambda}({x_{t})}}

        for t\geq 1, where \mathbf{x_{0}} is the initial vector, and \alpha, \beta,
        are the parameters of the algorithm.

        Parameters
        ----------
        lmb :   float
                The threshold value to compute the operator.
        alpha : float
                Convergence parameter
        beta :  float
                Convergence parameter
        max_itr : int
                Maximum number of iterations
        '''
        y = self.measurements()

        print('---------TwIST method---------- \n')

        dim = self.x.shape
        #x = np.zeros((635835, 1)) #check sizes
        x = self.A.transpose(np.zeros(y.shape))
        test = self.A.direct(x)
        hist = np.zeros((max_itr + 1, 4))

        residualx = 1
        tol = 1e-3

        nz_x = (x != 0.0)*1.0
        num_nz_x = np.sum(nz_x)

        #print('itr \t ||x-xold|| \t PSNR \n')
        itr = 0
        x_old = x
        IST_iters = 0
        TwIST_iters = 0
        xm2 = x
        xm1 = x
        max_svd = 1
        resid = y - self.A.direct(x)
        verbose = True
        enforceMonotone = True
        prev_f = 0.5 * np.sum(resid * resid) #"+ lmb * np.sum(np.abs(xm2))
        lm1 = 0.1
        rho0 = (1 - lm1 / 1) / (1 + lm1 / 1)
        alpha = 2 / (1 + np.sqrt(1 - rho0 ** 2))
        beta = alpha * 2/(lm1+1)
        while itr < max_itr:
            grad = self.A.transpose(resid)
            while True:
                x = soft_threshold(xm1+(grad/max_svd), lmb/max_svd)
                if (IST_iters >= 2) or (TwIST_iters != 0):
                    #for sparse
                    mask = (x != 0) * 1.0
                    xm1 = xm1 * mask
                    xm2 = xm2 * mask
                    #end for sparse
                    xm2 = (alpha-beta)*xm1 + (1-alpha)*xm2 + beta*x
                    resid = y - self.A.direct(x)
                    f = 0.5 * np.sum(resid * resid) + lmb * np.sum(np.abs(xm2))
                    if f > prev_f and enforceMonotone:
                        TwIST_iters = 0
                    else:
                        TwIST_iters = TwIST_iters + 1
                        IST_iters = 0
                        x = xm2
                        if TwIST_iters % 10000 == 0:
                            max_svd = 0.9 * max_svd
                        break
                else:
                    resid = y - self.A.direct(x)
                    f = 0.5 * np.sum(resid * resid) + lmb * np.sum(np.abs(xm2))
                    if f > prev_f:
                        max_svd = 1.5 * max_svd
                        if verbose:
                            print("Increasing S={mx: 2.2f}\n".format(mx=max_svd))
                            IST_iters = 0
                            TwIST_iters = 0
                        break
                    else:
                        TwIST_iters = TwIST_iters + 1
                        break

            xm2 = xm1
            xm1 = x
            nz_x_prev = nz_x
            nz_x = (x != 0.0)*1
            num_nz_x = np.sum(nz_x)
            num_changes_active = (np.sum(nz_x != nz_x_prev))
            itr = itr + 1
            prev_f = f
            residualx = np.linalg.norm(x - x_old) / np.linalg.norm(x)
            psnr_val = PSNR(self.x[:, :], np.transpose(self.operator_inv(x))[:, :])#self.H_elim
            ssim_val = ssim(self.x[:, :], np.transpose(self.operator_inv(x))[:, :])
            tv_val = tv_norm(self.operator_inv(x))

            hist[itr, 0] = residualx
            hist[itr, 1] = psnr_val
            hist[itr, 2] = ssim_val
            hist[itr, 3] = tv_val

            print(itr, '\t Error:', format(hist[itr, 0], ".2e"), '\t PSNR:', format(hist[itr, 1], ".3f"), 'dB',
                  '\t SSIM:', format(hist[itr, 2], ".3f"), '\t TV norm: ', format(hist[itr, 3], ".2f"))
            print(" obj={obj: 2.2f}, nz_x={nz: d}\n".format( obj=f, nz=int(num_nz_x)))
            yield itr, dict(result=self.operator_inv(x), hist=hist)
        yield self.operator_inv(x), hist


    def ADMM(self, rho, gamma, lmb, max_itr):
        # '''
        # The alternating direction method of multipliers (ADMM) is an algorithm that solves convex optimization problems
        # by using a divide and conquer strategy, breaking the problem into small pieces which are easier to handle.
        # The standard optimization formulation problem for the ADMM algorithm is defined as:
        #
        # .. math::
        #     \underset{\mathbf{x,z}}{\text{min }} \left\{ f(\mathbf{x}) + g(\mathbf{z}) \right\} \\
        #     \text{subject to} \mathbf{Ax + Bz = c}
        #
        # where f and g are closed, proper and convex functions. To simplify the algorithm usually is preferred that the g
        # function has a closet solution for the proximal operator.
        #
        # In the seismic reconstruction problem, the optimization problem associated could be defined as:
        #
        # .. math::
        #     \underset{\mathbf{x,v}}{\text{min }} \left\{ \frac{1}{2}\| \mathbf{y - H\Phi x} \|_2^2 + \lambda\|\mathbf{v}\|_1\right\} \\
        #     \text{subject to } \mathbf{x = v}
        #
        # where the first term of the cost function is a data fidelity term of the partially observed data, and the second
        # term promotes the smoothness of the coefficients of the recovered signal in a representation base.
        #
        # Parameters
        # ----------
        # rho :   float
        #         The weight for the dual problem term.
        # gamma :   float
        #         A relaxation coefficient for the dual problem parameter.
        # lmb :   float
        #         The threshold value to compute the operator.
        # max_itr : int
        #         Maximum number of iterations
        # '''
        s = 0

        y = self.measurements()

        #print('---------ADMM method---------- \n')

        hist = np.zeros((max_itr + 1, 4))
        dim = self.x.shape
        x = np.zeros(dim)

        begin_time = time.time()

        residualx = 1
        tol = 1e-3

        v = np.zeros(dim)
        u = np.zeros(dim)
        #print('itr \t ||x-xold|| \t PSNR \n')
        itr = 0

        Ht = self.H.transpose()
        HtY = Ht * np.squeeze(y.T.reshape(-1))  # H' * x
        HtY = np.reshape(HtY, [self.m, self.n], order='F')

        HTH = self.H.transpose() * self.H
        I_d = scipy.sparse.eye(HTH.shape[0])

        import matplotlib.pyplot as plt

        while (itr < max_itr):  # & residualx <= tol):
            x_old = x

            # F-update
            Inve = HTH + rho * I_d
            b = scipy.sparse.find(Inve)
            val = 1 / b[2]
            Inve = csr_matrix((val, (b[0], b[1])), shape=Inve.shape)

            x = HtY + rho * self.operator_dir(v - u)
            x = Inve * (x.T.reshape(-1))
            x = np.reshape(x, [self.m, self.n], order='F')

            # Proximal
            vtilde = self.operator_inv(x) + u
            v = soft_threshold(vtilde, lmb / rho)
            # Update langrangian multiplier
            u = vtilde - v

            # update rho
            rho = rho * gamma
            itr += 1
            residualx = np.linalg.norm(x - x_old) / np.linalg.norm(x)

            # psnr_val = PSNR(x, x_old)
            psnr_val = PSNR(self.x[:, self.H_elim], x[:, self.H_elim])
            ssim_val = ssim(self.x[:, self.H_elim], x[:, self.H_elim])
            tv_val = tv_norm(self.operator_inv(x))

            hist[itr, 0] = residualx
            hist[itr, 1] = psnr_val
            hist[itr, 2] = ssim_val
            hist[itr, 3] = tv_val

            if (itr + 1) % 5 == 0:
                # mse = np.mean(np.sum((y-A(v,Phi))**2,axis=(0,1)))
                end_time = time.time()
                # Error = %2.2f,
                #print("ADMM-TV: Iteration %3d,  Error = %2.2f, PSNR = %2.2f dB, SSIM = %1.2f, TV norm = %4.2f time = %3.1fs." % (
                #    itr + 1, residualx, psnr_val, ssim_val, tv_val, end_time - begin_time))

            yield itr, dict(result=x, hist=hist)

        yield x, hist