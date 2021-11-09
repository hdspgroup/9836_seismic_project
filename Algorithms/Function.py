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
# from skimage.measure import (compare_psnr, compare_ssim)


def dct2():
    '''
    This is a Discrete Cosine Transform for 2D signals
    (completar con una descripcion detallada en ingles)
    '''
    def dct2_function(x):
        return scipy.fft.dct(scipy.fft.dct(x.T).T)
    return  dct2_function

def idct2():
    def idct2_function(x):
        return scipy.fft.idct(scipy.fft.idct(x.T).T)
    return idct2_function

class Operator:
  def __init__(self, H, m,n,operator_dir,operator_inv):
    self.H = H
    self.m = m
    self.n = n
    self.operator_dir = operator_dir
    self.operator_inv = operator_inv

  def transpose(self,x):  # y = D'H' * x
      y = self.H.T * np.squeeze(x.reshape(-1))  # H' * x
      y = np.reshape(y, [self.m, self.n])
      y = self.operator_dir(y)
      return  y

  def direct(self,x): # y = H * D * x
      x = np.reshape(x, [self.m, self.n])  # ordenar
      theta = self.operator_inv(x)  # D * x
      y = self.H * np.squeeze(theta.reshape(-1))  # H * D * x
      return  y




# -------------------------------------------------------------------------
def soft_threshold(x,t):
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
    def __init__(self, x, H, operator_dir, operator_inv):

        # ------- change the dimension of the inputs image --------
        m, n = x.shape
        m = int(2 ** (np.ceil(np.log2(m)) - 1))
        n = int(2 ** (np.ceil(np.log2(n)) - 1))
        x = transform.resize(x, (m, n))
        x = x / np.abs(x).max()
        self.x = x
        self.m, self.n = x.shape

        # ---------- Load or build the sensing matrix -------------
        if (np.sum(H) > 1):
            self.H = H
        else:
            Nsub = int(np.round(m * (H)))
            # iava = np.random.permutation(m)
            iava = np.squeeze(loadmat('data/iava.mat') ['iava']) - 1
            self.cort = np.sort(iava [Nsub:])
            iava = np.sort(iava [0:Nsub])
            temp = np.asarray(range(0, int(m * n), m))
            temp = np.tile(temp, (len(iava), 1))
            temp = np.expand_dims(iava.T, -1) + temp
            temp = np.squeeze(temp.T.reshape(-1))
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
        :return: measures Y
        '''

        return self.H * np.squeeze(self.x.reshape(-1))

    def FISTA(self, lamnda, mu, max_itr):

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
            x = soft_threshold(z, lamnda)

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
    def GAP(self, lamnda, max_itr): # 311021

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
            x = soft_threshold(z, lamnda)
            itr = itr + 1

            residualx = np.linalg.norm(x - x_old) / np.linalg.norm(x)

            psnr_val = PSNR(self.operator_inv(x), self.x)

            hist[itr, 0] = residualx
            hist[itr, 1] = psnr_val

            print(itr, '\t Error:', format(hist[itr, 0], ".4e"), '\t PSNR:', format(hist[itr, 1],".3f"), 'dB \n')

        return self.operator_inv(x), hist

    #---------------------------------------------
    def TwIST(self, lamnda, alpha, beta, max_itr):

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
            s = soft_threshold(z, lamnda)

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