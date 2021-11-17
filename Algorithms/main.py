import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from Function import *
from skimage.metrics import structural_similarity as ssim

import scipy
#----------------- --------------------
x = np.load('data/data.npy')
x = x.T
maxiter = 1000


'''
---------------  SAMPLING --------------------
'''
sr_rand = 0.5 # 1-compression
y_rand, pattern_rand, pattern_index= random_sampling(x,sr_rand)
H = pattern_index

'''
---------------- RECOVERY ALGORITHM -----------------
Select the Algorithm: FISTA , GAP , TWIST , ADMM
'''
case = 'FISTA'
#----------------- FISTA ------------------------------
if case == 'FISTA':
    Alg = Algorithms(x, H , 'DCT2D', 'IDCT2D')
    tau = 0.1
    mu = 0.3
    x_result, hist = Alg.FISTA(tau, mu, maxiter)

# ------------------GAP--------------
if case == 'GAP':
    Alg = Algorithms(x,H, 'DCT2D', 'IDCT2D')
    tau = 1e-0
    x_result, hist = Alg.GAP(tau, maxiter) # inputs: tau, maxiter

# ------------------TwIST--------------
if case == 'TWIST':
    Alg = Algorithms(x,H, 'DCT2D', 'IDCT2D')
    # default parameters
    alpha = 1.2
    beta  = 1.998
    tau = 0.5

    x_result, hist = Alg.TwIST(tau, alpha, beta, maxiter)

# --------------- ADMM -----------------
if case == 'ADMM':
    Alg = Algorithms(x,H, 'DCT2D', 'IDCT2D')
    # default parameters
    # step_size = 1e-2
    # weight  = 0.5
    # eta = 1e-1
    rho = 0.5
    gamma = 1
    lmb = 0.0078

    x_result, hist = Alg.ADMM(rho, gamma, lmb, maxiter)
# --------------------------------

#-------------- Visualization ----------------

temp = np.asarray(range(0, pattern_rand.shape[0]))
pattern_rand_b2 = np.asarray(pattern_rand, dtype=bool) == 0
H_elim = temp[pattern_rand_b2]


x = Alg.x

fig, axs = plt.subplots(2, 2,dpi=150)
fig.suptitle('Results from the ' + case + ' Algorithm')

axs[0, 0].imshow(x, cmap='seismic', aspect='auto')
axs[0, 0].set_title('Real data')

axs[1, 0].imshow(y_rand, cmap='seismic', aspect='auto')
axs[1, 0].set_title('Incomplete')

# axs[1, 0].sharex(axs[0, 0])
metric = PSNR(x[:, H_elim],x_result[:, H_elim])
metric_ssim = ssim(x[:, H_elim],x_result[:, H_elim])
axs[0, 1].imshow(x_result, cmap='seismic', aspect='auto')
axs[0, 1].set_title('Reconstructed (PSNR:' + str("{:.1f})".format(metric)))
print(metric_ssim)
index = 1
axs[1, 1].plot(x[:, H_elim[index]], 'r', label='Reference')
axs[1, 1].plot(x_result [:, H_elim[index]], 'b', label='Recovered')
axs[1, 1].legend(loc='best')
plt.title('Trace ' + str("{:.0f}".format(H_elim[index])))

fig.tight_layout()
plt.show()
