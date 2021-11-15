import scipy.io
import  numpy as np
from skimage import transform
from scipy.io import loadmat
from scipy.sparse import csr_matrix
import inspect
import scipy.sparse.linalg as ln
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import time

def random_sampling(x,sr):
  ''' Random Sampling
  x : full data
  sr: subsampling factor
  '''
  dim = x.shape
  batch = 1 #dim[0]
  M  = dim[0]
  N  = dim[1]
  # L = dim[3]

  tasa_compression = int(sr*N)
  pattern_vec = np.ones((N,))

  ss = np.random.permutation(list(range(1, N-1)))
  pattern_vec[ss[0:tasa_compression]] = 0
  H0 = np.tile(pattern_vec.reshape(1,-1),(M,1))

  out = x * H0

  pattern_bool = np.asarray(pattern_vec, dtype=bool)


  return out,pattern_vec, pattern_bool

def dct2():
    '''
    This is a Discrete Cosine Transform for 2D signals
    Karen added this comment :)
    '''
    def dct2_function(x):
        return (scipy.fft.dct(scipy.fft.dct(x).T)).T
    return  dct2_function

def idct2():
    ''' Inverse operator of the 2D discrete cosine transform'''
    def idct2_function(x):
        return (scipy.fft.idct(scipy.fft.idct(x).T)).T
    return idct2_function

class Operator:
  def __init__(self, H, m, n, operator_dir,operator_inv):
    self.H = H
    self.m = m
    self.n = n
    self.operator_dir = operator_dir
    self.operator_inv = operator_inv

  def transpose(self,x):  # y = D'H' * x
      Ht = self.H.transpose()
      y = Ht* np.squeeze(x.T.reshape(-1))  # H' * x

      y = np.reshape(y, [self.m, self.n],order='F')

      y = self.operator_dir(y)

      return  y

  def direct(self,x): # y = H * D * x
      x = np.reshape(x, [self.m, self.n],order='F')  # ordenar

      theta = self.operator_inv(x)  # D * x

      y = self.H * np.squeeze(theta.T.reshape(-1))  # H * D * x

      return  y

# -------------------------------------------------------------------------
def soft_threshold(x,t):
    '''
    This is implementation of a sof-thresholding operator
    '''
    tmp = (np.abs(x)-t)
    tmp = (tmp+np.abs(tmp))/2
    y   = np.sign(x)*tmp
    return  y

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = np.max(original)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


class Algorithms:
    '''
    This is the main class of Function, which contain the optimization algorithms

    Input:
          x               The image to be sampled
          H:              The sensing matrix or the trace position to be deleted
          operator_dir :  The name of the sparsity direct transform or a function with the transform
          operator_inv :  The name of the sparsity inverse transform or a function with the inverse transform
    '''
    def __init__(self, x, H, operator_dir, operator_inv):

        # ------- change the dimension of the inputs image --------
        m, n = x.shape
        #m = int(2 ** (np.ceil(np.log2(m)) - 1))
        #n = int(2 ** (np.ceil(np.log2(n)) - 1))
        x = transform.resize(x, (m, n))
        x = x / np.abs(x).max()
        self.x = x
        self.m, self.n = x.shape
        self.pattern = 0

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
            iava = np.squeeze(loadmat('data/iava.mat') ['iava']) - 1
            self.cort = np.sort(iava [Nsub:])
            iava = np.sort(iava [0:Nsub])
            self.pattern = iava
            temp0 = np.reshape(np.asarray(range(0, m * n)), [n, m]).T
            temp = temp0[:, iava]
            temp = temp.T.reshape(-1)  # Column Vectorization
            self.H = csr_matrix((np.ones(temp.shape), (range(0, len(temp)), temp)), shape=(len(temp), int(m * n)))


        # ---------- Load or create the basis function  ---------

        if (inspect.isfunction(operator_dir)):
            self.operator_dir = operator_dir
        else:
            if operator_dir == 'DCT2D':
                self.operator_dir = dct2()

        if (inspect.isfunction(operator_inv)):
            self.operator_inv = operator_inv
        else:
            if operator_inv == 'IDCT2D':
                self.operator_inv = idct2()

        # ------------ This is a special class of operator ------------
        self.A = Operator(self.H, self.m, self.n, self.operator_dir, self.operator_inv)

    def measurements(self):
        '''
        Operator measurement models the subsampled acquisition process given a
        sampling matrix H
        :return: measures Y = H@x
        '''

        return self.H * np.squeeze(self.x.T.reshape(-1))


    # ---------------------------------------------FISTA----------------------------------------

    def FISTA(self, lmb, mu, max_itr):


        '''
       This is the python implementation of the FISTA (A Fast Iterative Shrinkage-Thresholding Algorithm )
       Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM journal on imaging sciences, 2(1), 183-202.
       Implemented by Jorge Bacca, Nov 2021, (jorge.bacca1@correo.uis.edu.co)

    Input:
          self            They have the variables of the Algorithm class, such as H,y, sparsity basis.
          lmb:            Is the sparsity regularizer
          mu :            Is the step-descent of the algorithm
          max_itr :       Is the maximum number of iterations
    '''

        y = self.measurements()

        print(' FISTA: \n')

        dim = self.x.shape
        x = np.zeros(dim)
        q = 1
        s = x
        hist = np.zeros((max_itr + 1, 2))
        print('itr \t ||x-xold|| \t PSNR \n')
        itr = 0
        while (itr < max_itr):
            x_old = x
            s_old = s
            q_old = q

            grad = self.A.transpose(self.A.direct(s_old) - y)  # Ht * (H * s_old - y)
            z = s_old - mu * (grad)

            # proximal
            x = soft_threshold(z, lmb)

            q = 0.5 * (1 + np.sqrt(1 + 4 * (q_old ** 2)))
            s = x + ((q_old - 1) / (q)) * (x - x_old)
            itr = itr + 1

            residualx = np.linalg.norm(x - x_old) / np.linalg.norm(x)

            psnr_val = PSNR(self.operator_inv(s), self.x)

            hist [itr, 0] = residualx
            hist [itr, 1] = psnr_val

            print(itr, '\t Error:', format(hist[itr, 0], ".4e"), '\t PSNR:', format(hist[itr, 1],".3f"), 'dB \n')

        return self.operator_inv(s), hist

    # ---------------------------------------------GAP----------------------------------------
    def GAP(self, lmb, max_itr):

        y = self.measurements()

        print('---------GAP method---------- \n')

        dim = self.x.shape
        x = np.zeros(dim)
        hist = np.zeros((max_itr + 1, 2))

        residualx = 1
        tol = 1e-2

        print('itr \t ||x-xold|| \t PSNR \n')
        itr = 0
        while (itr < max_itr ): #& residualx>tol):
            x_old = x

            temp = self.A.direct(x) - y

            grad = self.A.transpose(temp)
            z = x - grad

            # proximal
            x = soft_threshold(z, lmb)
            itr = itr + 1

            residualx = np.linalg.norm(x - x_old) / np.linalg.norm(x)

            psnr_val = PSNR(self.operator_inv(x), self.x)

            hist[itr, 0] = residualx
            hist[itr, 1] = psnr_val

            print(itr, '\t Error:', format(hist[itr, 0], ".4e"), '\t PSNR:', format(hist[itr, 1],".3f"), 'dB \n')

        return self.operator_inv(x), hist

    #----------------TWIST----------------------------
    def TwIST(self, lmb, alpha, beta, max_itr):

        y = self.measurements()

        print('---------TwIST method---------- \n')

        dim = self.x.shape
        x = np.zeros(dim)
        hist = np.zeros((max_itr + 1, 2))

        residualx = 1
        tol = 1e-3

        print('itr \t ||x-xold|| \t PSNR \n')
        itr = 0
        x_old = x

        while (itr < max_itr ): #& residualx <= tol):

            temp = self.A.direct(x) - y

            grad = self.A.transpose(temp)
            z = x - grad

            # proximal
            s = soft_threshold(z, lmb)

            # Actualizacion
            temp = (1 - alpha)* x_old + (alpha - beta)* x + beta * s
            x_old = x
            x = temp

            itr = itr + 1

            residualx = np.linalg.norm(x - x_old) / np.linalg.norm(x)

            psnr_val = PSNR(self.operator_inv(x), self.x)

            hist[itr, 0] = residualx
            hist[itr, 1] = psnr_val

            print(itr, '\t Error:', format(hist[itr, 0], ".4e"), '\t PSNR:', format(hist[itr, 1],".3f"), 'dB \n')

        return self.operator_inv(x), hist

    def ADMM(self, rho, gamma, lamnda, max_itr):


        s=0


        y = self.measurements()

        print('---------ADMM method---------- \n')

        hist = np.zeros((max_itr + 1, 2))
        dim = self.x.shape
        x = np.zeros(dim)

        begin_time = time.time()

        residualx = 1
        tol = 1e-3

        v = x.ravel()
        u = x.ravel()

        print('itr \t ||x-xold|| \t PSNR \n')
        itr = 0

        HtY = self.A.transpose(y)

        while (itr < max_itr ): #& residualx <= tol):
            x_old = x

            # F-update
            temp = HtY.ravel() + np.sqrt(rho)*(v-u)
            x = ln.cgs(self.A, temp, x0=None, tol=1e-05, maxiter=20)

            # Proximal
            vtilde = x + u
            v = soft_threshold(vtilde, lamnda/rho)

            # Update langrangian multiplier
            u = u + (x - v)

            # update rho
            rho = rho * gamma

            itr+=1

            residualx = np.linalg.norm(x - x_old)  / np.linalg.norm(x)

            x = np.reshape(x, [self.m, self.n])
            psnr_val = PSNR(self.operator_inv(x), x_old)
            hist[itr, 0] = residualx
            hist[itr, 1] = psnr_val

            if (itr + 1) % 5 == 0:
                # mse = np.mean(np.sum((y-A(v,Phi))**2,axis=(0,1)))
                end_time = time.time()
                # Error = %2.2f,
                print("ADMM-TV: Iteration %3d,  Error = %2.2f, PSNR = %2.2f dB," 
                      " time = %3.1fs."
                      % (itr + 1, residualx, psnr_val, end_time - begin_time))
                      #% (ni + 1, psnr(v, X_ori), end_time - begin_time))

        return self.operator_inv(v),hist
