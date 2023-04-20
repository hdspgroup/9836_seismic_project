import os.path
from keras.callbacks import Callback
import tensorflow as tf
import numpy as np

path_results = os.path.join("..", "Results2D")

def myprint(s):
    with open('modelsummary.txt','w+') as f:
        print(s, file=f)

def reduce_lr():
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_psnr', factor=0.98,
                                                 patience=100, min_lr=1e-5, mode='max')
    return reduce_lr

def mcp_save(path_results):

    save_path = os.path.join(path_results, 'weights_best.h5')
    mcp_save = tf.keras.callbacks.ModelCheckpoint(save_path, monitor='loss', verbose=1,
                                              save_best_only=False, save_weights_only=True, mode='min',
                                              save_freq='epoch')
    return mcp_save

def normalize_data(data):
    # data2 = data + np.abs(data.min())
    data2 = data / data.max()
    return data2

def print_SR(weights, data):
    return print('non-removed traces: ' + str(tf.math.reduce_sum(weights)) +
          f'\nsampling rate is: ' + str(tf.math.reduce_sum(weights) * 100 / data.shape[2]))