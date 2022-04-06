import numpy as np
from numpy.linalg import inv as inv
import numpy.linalg as ng
import imageio
import matplotlib.pyplot as plt
from Function import *
import time


def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order='F'), 0, mode)


def soft_thresholding(mat, lambda0):  ## Nuclear norm regularization (convex)
    u, s, v = np.linalg.svd(mat, full_matrices=0)
    vec = s - lambda0
    pos = np.where(vec < 0)
    vec[pos] = 0

    return np.matmul(np.matmul(u, np.diag(vec)), v)


def GLTC(dense_tensor, sparse_tensor, alpha, beta, rho, maxiter):
    dim0 = sparse_tensor.ndim
    dim1, dim2, dim3 = sparse_tensor.shape
    position = np.where(sparse_tensor != 0)
    binary_tensor = np.zeros((dim1, dim2, dim3))
    binary_tensor[position] = 1
    tensor_hat = sparse_tensor.copy()

    X = np.zeros((dim1, dim2, dim3, dim0))  # \boldsymbol{\mathcal{X}} (n1*n2*3*d)
    Z = np.zeros((dim1, dim2, dim3, dim0))  # \boldsymbol{\mathcal{Z}} (n1*n2*3*d)
    T = np.zeros((dim1, dim2, dim3, dim0))  # \boldsymbol{\mathcal{T}} (n1*n2*3*d)
    for k in range(dim0):
        X[:, :, :, k] = tensor_hat

    D1 = np.zeros((dim1, dim1))
    for i in range(dim1 - 1):
        D1[i + 1, i] = -1
        D1[i + 1, i + 1] = 1
    D2 = np.zeros((dim2, dim2))
    for i in range(dim2 - 1):
        D2[i + 1, i] = -1
        D2[i + 1, i + 1] = 1

    for iters in range(maxiter):
        for k in range(dim0):
            Z[:, :, :, k] = mat2ten(soft_thresholding(ten2mat(X[:, :, :, k] + T[:, :, :, k] / rho, k),
                                                      alpha / rho), np.array([dim1, dim2, dim3]), k)
            if k == 0:
                var0 = mat2ten(np.matmul(inv(beta * np.matmul(D1.T, D1) + rho * np.eye(dim1)),
                                         ten2mat(rho * Z[:, :, :, k] - T[:, :, :, k], k)),
                               np.array([dim1, dim2, dim3]), k)
            elif k == 1:
                var0 = mat2ten(np.matmul(inv(beta * np.matmul(D2.T, D2) + rho * np.eye(dim2)),
                                         ten2mat(rho * Z[:, :, :, k] - T[:, :, :, k], k)),
                               np.array([dim1, dim2, dim3]), k)
            else:
                var0 = Z[:, :, :, k] - T[:, :, :, k] / rho
            X[:, :, :, k] = (np.multiply(1 - binary_tensor, var0)
                             + np.multiply(binary_tensor, sparse_tensor))
        tensor_hat = np.mean(X, axis=3)
        for k in range(dim0):
            var = T[:, :, :, k] + rho * (X[:, :, :, k] - Z[:, :, :, k])
            T[:, :, :, k] = var.copy()
            X[:, :, :, k] = tensor_hat.copy()

    return tensor_hat


if __name__ == '__main__':
    alpha = 10
    rho = 1
    beta = 0.1 * rho
    maxiter = 1000

    data_name = 'cube4.npy'

    x = np.load('../data/' + data_name)

    if data_name == 'data.npy':
        x = x.T
    x = x / np.abs(x).max()

    '''
    ---------------  SAMPLING --------------------
    '''
    sr_rand = 0.5  # 1-compression
    y_rand, pattern_rand, pattern_index = random_sampling(x[:, int(x.shape[1] / 2), :], sr_rand)
    H = pattern_index
    y = x.copy()
    y[..., pattern_rand == 0] = 0

    # y = np.reshape(y, [x.shape[0] * x.shape[1], x.shape[2]])
    # x = np.reshape(x, [x.shape[0] * x.shape[1], x.shape[2]])
    y = np.transpose(y, [0, 2, 1])
    x = np.transpose(x, [0, 2, 1])

    start = time.time()
    image_hat = GLTC(x, y, alpha, beta, rho, maxiter)
    end = time.time()
    print(f"Duration: {end - start}")
    image_rec = np.round(image_hat).astype(int)
    image_rec[np.where(image_rec > 255)] = 255
    image_rec[np.where(image_rec < 0)] = 0
    pos = np.where((x != 0) & (y == 0))
    rse = np.linalg.norm(image_rec[pos] - x[pos], 2) / np.linalg.norm(x[pos], 2)
