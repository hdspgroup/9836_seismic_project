import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
from SensingModel2D import *

def min_variance(y,param,model):
    y = tf.reduce_sum(y,axis=[1,2,3])
    y /= tf.reduce_max(y)
    var_loss =   tf.math.reduce_std(tf.math.reduce_std(y,0))+tf.keras.backend.epsilon()
    model.add_loss(var_loss*param)
    model.add_metric(var_loss,'var_min_loss')
def laplacian_kl(y,stddev,mean,model):
    y = tf.reduce_mean(y, axis=[1, 2, 3])

    y /= tf.reduce_max(y)
    z_mean = tf.reduce_mean(y+tf.keras.backend.epsilon(),0)
    z_var = (tf.math.reduce_std(y,0))+tf.keras.backend.epsilon()
    lp_kl = tf.reduce_mean((z_var*tf.math.exp(-tf.abs(z_mean-mean)/z_var)+tf.abs(z_mean-mean))/stddev+tf.math.log(stddev/(z_var+1e-6))-1)
    model.add_loss(lp_kl)
    model.add_metric(lp_kl, name='lp_kl_loss', aggregation='mean')

def variational(y,stddev,mean,model):
    y = tf.reduce_mean(y, axis=[1, 2, 3])
    y /= tf.reduce_max(y)
    z_mean = tf.reduce_mean(y+tf.keras.backend.epsilon(),0)
    z_log_var = tf.math.log(tf.math.reduce_std(y+tf.keras.backend.epsilon(),0))
    # tf.print(z_log_var.max())

    kl_loss = -0.5 * tf.math.reduce_mean(z_log_var - tf.math.log(stddev) - (tf.exp(z_log_var) + tf.pow(z_mean - mean, 2)) / (stddev ** 2) + 1)
    model.add_loss(kl_loss)
    model.add_metric(kl_loss, name='kl_loss', aggregation='mean')



def sensing_model(latent_dim, trainable):
    model = tf.keras.Sequential([
      MaskLayer_latendim(latent_dim, trainable=trainable),
    ], name='mask_model')
    return model

def build_model(recons_net, sensing_model, input_size,regularization,mean,stddev,type_reg,param):
    print(mean,stddev)
    inputs = Input(input_size)
    corrupted = sensing_model(inputs)
    outputs = recons_net(corrupted)
    model = Model(inputs, outputs)

    if regularization:
        if type_reg=='kl-gaussian':
            variational(model=model,y=corrupted,mean=mean,stddev=stddev)
        elif type_reg =='kl-laplacian':
            laplacian_kl(model=model,y=corrupted,mean=mean,stddev=stddev)
        elif type_reg=='min-variance':
            min_variance(model=model,y=corrupted,param=param)
    return model

def build_model2(recons_net, input_size):
    inputs = Input(input_size)
    corrupted = sensing_model(inputs)
    outputs = recons_net(corrupted)
    return Model(inputs, outputs)