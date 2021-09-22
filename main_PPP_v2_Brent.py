from __future__ import print_function
from builtins import input, range

import numpy as np
from scipy.sparse.linalg import LinearOperator

from bm3d import bm3d_rgb
from bm3d import bm3d

from sporco.linalg import _cg_wrapper
from ppp import PPPConsensus
from sporco.interp import bilinear_demosaic
from sporco import metric
from sporco import util
from sporco import plot
from sporco.prox import prox_l1
from scipy.io import loadmat
from functools import partial
import multiprocessing as mp
from scipy.io import savemat
import pywt
from skimage import transform
import numpy as np

from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt


def fk(data,dt,dx):
    nt=data.shape[0]
    nx=data.shape[1]
    nt_fft=2*nt
    nx_fft=2*nx
    data_f=np.fft.fft(data,n=nt_fft,axis=0)
    data_fk=np.fft.fft(data_f,n=nx_fft,axis=1)
    FK=20*np.log10(np.fft.fftshift(np.abs(data_fk)))
    FK=FK[nt:,:]
    f = np.linspace(-0.5,0.5,nt_fft)/dt
    kx = np.linspace(-0.5,0.5,nt_fft)/dx
    return FK, f, kx

def PSSIFK(ref,comp):
    ''' computes PSNR and SSIM in FK domain'''
    ref_fk = fk(ref,0.003,25)[0]
    comp_fk = fk(comp,0.003,25)[0]
    pssnr = metric.psnr(ref_fk,comp_fk)
    #sssim = ssim(ref_fk,comp_fk)
    return (pssnr,0)

def fkk(indxx, imgp):
    data_fk, ff, kx = fk(img[:, :, shots_del[indxx]], 0.003, 25)
    data_fk3, ff3, kx3 = fk(imgp[:, :, shots_del[indxx]], 0.003, 25)
    psnr3 = metric.psnr(data_fk, data_fk3)

    plt.subplot(121)
    plt.imshow(data_fk, aspect='auto', cmap='jet',
               extent=[kx.min(), kx.max(), ff.max(), 0])
    plt.colorbar(orientation="horizontal", pad=0.1)
    plt.title(f'Ground Truth {shots_del[indxx] + 1}')
    plt.subplot(122)
    cp = plt.imshow(data_fk3, aspect='auto', cmap='jet',
                    extent=[kx.min(), kx3.max(), ff3.max(), 0])
    plt.colorbar(orientation="horizontal", pad=0.1)
    cp.set_clim([data_fk.min(), data_fk.max()])
    plt.title('Recovered wavelet cool \n PSNR %.2f (dB) ' % (psnr3), fontsize=10, color='blue')
    plt.show()


def normalize_data(data):
    min_data  = data.min()
    scale     = np.max(data-min_data)
    norm_data = (data-min_data)/scale
    return norm_data, min_data,scale

def A(x,shots_sam):
    """ Forward model
    """
    y = np.zeros((x.shape))
    y[:, shots_sam,:] = x[:,shots_sam,:]
    return y

def AT(x,shots_sam):
    """Back projection
    """
    y = np.zeros(x.shape)
    y[:, shots_sam, :]= x[:, shots_sam, :]
    return y

def bm3d_map(x,bsigma):
    return bm3d(x, bsigma)

def sninterp(data, sre):
    '''
    basic interpolation between shots
    sre = missing sources index - list
    data = seismic cube with missin sources in zero - array
    '''
    np.random.seed(12345)
    data = data + np.random.normal(0.0, 0.5, data.shape).astype(np.float32)
    data = data.transpose([0, 2, 1])
    for i in sre:
        if i == data.shape[2]:
            data[:, :, i] = data[:, :, i - 1] / 2
        elif i == 0:
            data[:, :, i] = data[:, :, i + 1] / 2
        else:
            data[:, :, i] = (data[:, :, i + 1] + data[:, :, i - 1]) / 2
    return data.transpose([0, 2, 1])

''' Loading data'''
img = np.load('data/cube4.npy')[:700,:,:10]
# plt.imshow(img[:,:,0],aspect='auto',interpolation='bicubic',cmap='gray')
img   = img.transpose([0,2,1]) # (time,sources,receivers)

"""
Shots to be removed stored in variable --> shots_del
Load mat file or include the shots manually in shots_del
"""
#random
np.random.seed(0)
# idx = np.random.permutation(img.shape[1])
# cr    = 0.2  # compressive ratio
# shots = np.round(cr * img.shape[1])
# shots = shots.astype(int) # shots deleted
# shots_del = idx[:shots]
# shots_sam = idx[shots:]

# Manually
idx        = np.arange(img.shape[1])
shots_del  = [2,5,8]
shots_sam  = list(set(idx)-set(shots_del))
print(shots_del)
""""""""""""""""""

s       = A(img,shots_sam)   # Measurements (Subsampled shots)
imgshp  = img.shape       # Shape of reconstructed RGB image
imgsz   = img.size        # Size of reconstructed RGB image
sn      = s
sini    = sninterp(s.copy(), shots_del)

def f(x):
    return 0.5 * np.linalg.norm((A(x,shots_del) - sn).ravel())**2

def proxf(x, rho, tol=1e-3, maxit=50):
    ATA  = lambda z: AT(A(z,shots_sam),shots_sam)
    ATAI = lambda z: ATA(z.reshape(imgshp)).ravel() + rho * z.ravel() 
    lop  = LinearOperator((imgsz, imgsz), matvec=ATAI, dtype=s.dtype)
    b    = AT(sn,shots_sam) + rho * x
    vx, cgit = _cg_wrapper(lop, b.ravel(), None, tol, maxit)
    return vx.reshape(imgshp)

bsigma1 = 5e-2 # Denoiser parameter
lmb     = 1e-3 # thresholding parameter

nproc = 8 # number of cores to use
if nproc is None:
    nproc = mp.cpu_count()

def proxg1_parallel(x,rho):
    bm3d_p = partial(bm3d_map, bsigma=bsigma1)
    pool = mp.Pool(processes=nproc)
    # parallel map returns a list
    res = pool.map(bm3d_p, (x[:, :, z] for z in np.arange(x.shape[2])))
    # copy the data to array
    for i in np.arange(0, x.shape[2]):
        x[:, :, i] = res[i]
    #x,aaa,bbb = normalize_data(x)
    return x
def proxg2_parallel(x,rho):
    x = x.transpose((2, 1, 0))
    bm3d_p = partial(bm3d_map, bsigma=bsigma1)
    pool = mp.Pool(processes=nproc)
    # parallel map returns a list
    res = pool.map(bm3d_p, (x[:, :, z] for z in np.arange(x.shape[2])))
    # copy the data to array
    for i in np.arange(0, x.shape[2]):
        x[:, :, i] = res[i]
    #x, aaa, bbb = normalize_data(x)
    return x.transpose((2,1,0))
def proxg_sparsity(x,rho):
    Nx, Ny, Nz  = x.shape
    Nxi         = 2**np.ceil(np.log2(Nx))
    Nyi         = 2**np.ceil(np.log2(Ny))
    Nzi         = 2**np.ceil(np.log2(Nz))
    x           = transform.resize(x,(Nxi,Nyi,Nzi))
    coeffs      = pywt.wavedecn(x, 'sym8', axes=(0, 1, 2))
    coeff_array, coeff_slices, coeff_shapes = pywt.ravel_coeffs(coeffs)
    coeff_array = prox_l1(coeff_array, lmb/rho)
    coeffs_rec  = pywt.unravel_coeffs(coeff_array, coeff_slices, coeff_shapes, output_format='wavedecn')
    data_rec    = pywt.waverecn(coeffs_rec, 'sym8')
    data_rec    = transform.resize(data_rec,(Nx,Ny,Nz))
    return data_rec

def proxg_sparsity_2(x,rho):
    Nx, Ny, Nz  = x.shape
    Nxi         = 2**np.ceil(np.log2(Nx))
    Nyi         = 2**np.ceil(np.log2(Ny))
    Nzi         = 2**np.ceil(np.log2(Nz))
    x           = transform.resize(x, (Nxi, Nyi, Nzi))
    coeffs      = pywt.wavedec2(x, 'sym8', axes=(0, 1))
    coeff_array, coeff_slices, coeff_shapes = pywt.ravel_coeffs(coeffs,axes=(0,1))
    coeff_array = prox_l1(coeff_array, lmb/rho)
    coeffs_rec  = pywt.unravel_coeffs(coeff_array, coeff_slices, coeff_shapes, output_format='wavedec2')
    data_rec    = pywt.waverec2(coeffs_rec, 'sym8',axes=(0,1))
    data_rec    = transform.resize(data_rec,(Nx,Ny,Nz))
    return data_rec




'''Sparsity only'''
# rho=0.008
# lmb=0.001
# opt = PPPConsensus.Options({'Verbose': True, 'RelStopTol': 1e-4,
#                     'MaxMainIter': 15, 'rho': rho, 'Y0': sn})

# b   = PPPConsensus(img.shape, (proxf, proxg_sparsity), opt=opt)

''' PPP with denoiser receiver slices '''

# rho=0.0001 # 0.008
# bsigma1=0.1 #0.001 init
# opt = PPPConsensus.Options({'Verbose': True, 'RelStopTol': 1e-4,
#                     'MaxMainIter': 15, 'rho': rho, 'Y0': sn})
# b   = PPPConsensus(img.shape, (proxf,proxg1_parallel), opt=opt)

''' # Sparsity + denoiser receiver slices '''
#rho=0.008
#bsigma1=0.05
#lmb=0.00001
#b   = PPPConsensus(img.shape, (proxf,proxg_sparsity), opt=opt)
''' Sparsity + denoiser recever slices + denoiser time slices'''
rho=0.008
bsigma1=0.001
lmb=0.00001
opt = PPPConsensus.Options({'Verbose': True, 'RelStopTol': 1e-4,
                    'MaxMainIter': 50, 'rho': rho, 'Y0': sn})
b   = PPPConsensus(img.shape, (proxf, proxg_sparsity, proxg1_parallel, proxg2_parallel), opt=opt)

imgp = b.solve()
np.save('CE_solution.npy',imgp.transpose([0,2,1]))

#img,aaa,bbb  = normalize_data(img)
#imgp,aaa,bbb = normalize_data(imgp)
print("PPP ADMM solve time:        %5.2f s" % b.timer.elapsed('solve'))
print("PPP seismic recovered PSNR:       %5.2f dB" % 
      PSSIFK(img[:,shots_del,:],imgp[:,shots_del,:])[0])


ss = 2
plt.figure(), 
plt.subplot(121)
plt.imshow(img[:,ss,:],aspect='auto',interpolation='bicubic',cmap='gray')
plt.subplot(122)
plt.imshow(imgp[:,ss,:],aspect='auto',interpolation='bicubic',cmap='gray')
print("PPP seismic recovered PSNR:       %5.2f dB" % 
      PSSIFK(img[:,ss,:],imgp[:,ss,:])[0])

#ssim(img[:,ss,:],imgp[:,ss,:])
