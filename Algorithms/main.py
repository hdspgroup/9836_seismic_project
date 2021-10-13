import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from Function import *



#----------------- Final shows --------------------
x = np.load('data/data.npy')
Alg = Algorithms(x, 0.75, 'DCT2D', 'IDCT2D')
x_result, hist = Alg.FISTA(0.1, 0.3,100)

x = Alg.x
cort = Alg.cort

plt.subplot(2, 2, 1), plt.imshow(x.T, cmap='gray', aspect='auto')
plt.title('Real data')
y_gor = x.copy()
y_gor[cort,:]= 0
plt.subplot(2, 2, 2), plt.imshow(y_gor.T, cmap='gray', aspect='auto')
plt.title('Input')
plt.subplot(2, 2, 3), plt.imshow(x_result.T, cmap='gray', aspect='auto')
plt.title('reconstruction')
plt.subplot(2, 2, 4), plt.plot(x [cort [1], :], 'r', label='Real')
plt.plot(x_result [cort [1], :], '-.b', label='Recovery')
plt.legend()
plt.show()

