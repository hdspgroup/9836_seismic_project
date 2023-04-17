import numpy as np
import random
import copy

def designed_sampling(x, pattern):
    '''
    x : full data
    patter: obtained pattern
    '''
    batch, M, N, L = x.shape
    out = []
    # train = []

    new = np.zeros_like(x)
    for i in range(batch):
        new[i, :, :, 0] = x[i, :, :, 0] * np.tile(pattern.reshape(1, -1), (x.shape[1], 1))
    out.append(new)

    return np.concatenate(out, axis=0)  # ,np.concatenate(train,axis=0)

def uniform_sampling(x, sr):
    '''
    Inputs:
    x : full data
    sr: subsampling factor

    Outputs:
    y : undesampled data
    pattern: pattern of subsampling [to be replicated at the row axis]
    '''
    L, M, N, batch = x.shape
    pattern_vec = np.ones((N,), dtype=int)
    n_col_rmv = np.round(N * sr)
    x_distance = np.round(N / n_col_rmv)

    i = 0
    while i * int(x_distance) < N:
        # for i in range(int(n_col_rmv)):
        pattern_vec[i * int(x_distance)] = 0
        i = i + 1

    out = []
    new = np.zeros_like(x)

    for i in range(L):
        new[i, :, :, 0] = x[i, :, :, 0] * np.tile(pattern_vec.reshape(1, -1), (M, 1))
    out.append(new)

    return np.concatenate(out, axis=0), pattern_vec  # np.concatenate(train,axis=0),

def jitter_sampling(x, sr, n_bloque):
    # https://slim.gatech.edu/Publications/Public/Journals/Geophysics/2008/hennenfent08GEOsdw/paper_html/node14.html
    '''
    Inputs:
    x : full data
    gamma: Undersampling factor [It should be odd, i.e, 1,3,5]

    Outputs:
    y : undesampled data
    pattern: pattern of subsampling [to be replicated at the row axis]
    '''
    L, M, N, batch = x.shape
    n = int(N * sr)
    samples_block = N / n_bloque # numero de trazas por bloque
    vec_total = []
    init = random.randint(samples_block * 0, samples_block * (0 + 1)-1)

    while len(vec_total) != n:
        for i in range(n_bloque):
            if len(vec_total) != n:
                while init in vec_total:
                    init = random.randint(samples_block * i, samples_block * (i + 1) - 1)
                vec_total.append(init)

    pattern_vec = np.ones(N)
    pattern_vec[np.array(vec_total)] = 0

    x2 = copy.deepcopy(x)
    x2[:,:,np.array(vec_total),:] = 0

    return x2, pattern_vec  # ,compression # np.concatenate(train,axis=0),

def random_sampling(x, sr):  # previous: samplingH
    '''
    x : full data
    sr: subsampling factor
    '''
    L, M, N, batch = x.shape
    out = []
    tasa_compression = int(sr * N)

    pattern_vec = np.ones((N,))

    ss = np.random.permutation(list(range(1, N - 1)))
    pattern_vec[ss[0:tasa_compression]] = 0

    new = np.zeros_like(x)
    for i in range(L):
        new[i, :, :, 0] = x[i, :, :, 0] * np.tile(pattern_vec.reshape(1, -1), (x.shape[1], 1))
    out.append(new)

    return np.concatenate(out, axis=0), pattern_vec  # np.concatenate(train,axis=0)