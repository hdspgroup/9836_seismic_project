import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from Function import *
import scipy
#----------------- --------------------
x = np.load('data/data.npy')
maxiter = 100

# Select the Algorithm
case = 3 # opt: 0,1,2
#----------------- FISTA --------------------
if case == 0:
    H = 0.75
    Alg = Algorithms(x,H , 'DCT2D', 'IDCT2D')

    tau = 0.1
    mu = 0.3
    x_result, hist = Alg.FISTA(tau, mu,maxiter)

# ------------------GAP--------------
if case == 1:
    H = 0.75
    Alg = Algorithms(x,H, 'DCT2D', 'IDCT2D')
    tau = 1e-0
    x_result, hist = Alg.GAP(tau, maxiter) # inputs: tau, maxiter

# ------------------TwIST--------------
if case == 2:
    H = 0.75
    Alg = Algorithms(x,H, 'DCT2D', 'IDCT2D')
    # default parameters
    alpha = 1.2
    beta  = 1.998
    tau = 0.5

    x_result, hist = Alg.TwIST(tau, alpha, beta, maxiter)

# --------------- ADMM -----------------
if case == 3:
    H = 0.75
    Alg = Algorithms(x,H, 'DCT2D', 'IDCT2D')
    # default parameters
    # step_size = 1e-2
    # weight  = 0.5
    # eta = 1e-1
    rho = 0.5
    gamma = 1
    lamnda = 0.0078

    x_result, hist = Alg.ADMM(rho, gamma, lamnda, maxiter)
# --------------------------------

#-------------- Visualization ----------------
x = Alg.x
cort = Alg.cort

plt.subplot(2, 2, 1), plt.imshow(x.T, cmap='gray', aspect='auto')
plt.title('Real data')
y_gor = x.copy()
y_gor[cort,:]= 0
plt.subplot(2, 2, 2), plt.imshow(y_gor.T, cmap='gray', aspect='auto')
plt.title('Input')
plt.subplot(2, 2, 3), plt.imshow(x_result.T, cmap='gray', aspect='auto')
plt.title('Reconstruction')
plt.subplot(2, 2, 4), plt.plot(x [cort [1], :], 'r', label='Reference')
plt.plot(x_result [cort [1], :], 'b', label='Recovered')
plt.legend()
plt.show()

