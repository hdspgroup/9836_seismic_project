import scipy.io
import  numpy as np
from skimage import transform
from scipy.io import loadmat
from scipy.sparse import csr_matrix
import inspect

def dct2():
    def dct2_function(x):
        return scipy.fft.dct(scipy.fft.dct(x.T).T)
    return  dct2_function

def idct2():
    def idct2_function(x):
        return scipy.fft.idct(scipy.fft.idct(x.T).T)
    return idct2_function

class Operator:
    """ Class with operators and metrics"""
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
    """
    Class with algorithms used in seismic reconstruction
    Solvers: FISTA, 
    """
    def __init__(self, x, H, operator_dir, operator_inv):

        # ------- change the dimention of the inputs image --------
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
            iava = np.squeeze(loadmat('iava.mat') ['iava']) - 1
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

            residualx = np.linalg.norm(self.x - x_old) / np.linalg.norm(self.x)

            psnr_val = PSNR(self.operator_inv(s), self.x)

            hist [itr, 0] = residualx
            hist [itr, 1] = psnr_val

            print(itr, '\t', hist [itr, 0], '\t', hist [itr, 1], '\n')

        return self.operator_inv(s), hist
