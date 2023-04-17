import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np

SR = 0.7
@tf.function
def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.)/2., 0., 1)

@tf.function
def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    a op that behave as f(x) in forward mode,
    but as g(x) in the backward mode.
    '''
    rounded = tf.math.round(x)
    return x + tf.stop_gradient(rounded-x)
    '''
    The neurons activations binarization function
    It behaves like the sign function during forward propagation
    And like:
      hard_tanh(x) = 2*hard_sigmoid(x)-1
    during back propagation
    '''

@tf.function
def binary_sigmoid_unit(x):
    return round_through(hard_sigmoid(x))

@tf.function
def sampling_rate(x, sr=SR, rho=0.1):
  return rho*tf.math.square(tf.reduce_mean(binary_sigmoid_unit(x)) - sr)

class MaskLayer_latendim(tf.keras.layers.Layer):
    def __init__(self, latent_dim, trainable):
        super(MaskLayer_latendim, self).__init__()
        self.latent = self.add_weight("latent", shape=(1, latent_dim),
                                      dtype='float32', trainable=trainable) # z

    def build(self, input_shape):
        # N , M , _ = input_shape
        self.mask_size = input_shape[1:]
        self.dense = Dense( self.mask_size[1], use_bias = False ) # A

    def get_weights(self):
        weights = self.dense(self.latent) # Az
        weights = tf.reshape(weights, [1, -1 , 1]) # [1 , M , 1]
        weights = tf.tile(weights, [self.mask_size[0], 1 , 1]) # [ N, M , 1]
        return weights # Mascara sin binarizar

    def call(self, inputs):
        weights = self.get_weights()
        mask = binary_sigmoid_unit(weights)
        x = tf.multiply(inputs, mask)
        self.add_loss(sampling_rate(binary_sigmoid_unit(weights)))
        return x

class MaskLayer(tf.keras.layers.Layer):
    def __init__(self, dims, trainable=True):
        super(MaskLayer, self).__init__()
        self.latent = self.add_weight("latent", shape=(1, dims),
                                      dtype='float32', trainable=trainable) # z

    def build(self, input_shape):
        # N , M , _ = input_shape
        self.mask_size = input_shape[1:]

    def get_weights(self):
        weights = self.latent
        weights = tf.reshape(weights, [1, -1 , 1]) # [1 , M , 1]
        weights = tf.tile(weights, [self.mask_size[0], 1 , 1]) # [ N, M , 1]
        return weights # Mascara sin binarizar

    def call(self, inputs):
        weights = self.get_weights()
        mask = binary_sigmoid_unit(weights)
        x = tf.multiply(inputs, mask)
        x = x*2-1
        self.add_loss(sampling_rate(binary_sigmoid_unit(weights)))
        return x