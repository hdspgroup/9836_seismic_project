import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers

def recons_network(DataTrain):
  input_size = DataTrain.shape[1:]
  inputs = tf.keras.Input(shape=input_size)

  c1 = Conv2D(16, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(inputs)
  c1 = Conv2D(16, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(c1)
  p1 = MaxPooling2D((2, 2))(c1)

  c2 = Conv2D(32, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(p1)
  c2 = Conv2D(32, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(c2)
  p2 = MaxPooling2D((2, 2))(c2)

  c3 = Conv2D(64, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(p2)
  c3 = Conv2D(64, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(c3)
  p3 = MaxPooling2D((2, 2))(c3)

  c4 = Conv2D(128, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(p3)
  c4 = Conv2D(128, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(c4)
  p4 = MaxPooling2D(pool_size=(2, 2))(c4)

  c5 = Conv2D(256, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(p4)
  c5 = Conv2D(256, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(c5)

  u6 = Conv2DTranspose(128, (2, 2),
                       strides=(2, 2),
                       padding='same')(c5)
  u6 = concatenate([u6, c4])
  c6 = Conv2D(128, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(u6)
  c6 = Conv2D(128, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(c6)

  u7 = Conv2DTranspose(64, (2, 2),
                       strides=(2, 2),
                       padding='same')(c6)
  u7 = concatenate([u7, c3])
  c7 = Conv2D(64, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(u7)
  c7 = Conv2D(64, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(c7)

  u8 = Conv2DTranspose(32, (2, 2),
                       strides=(2, 2),
                       padding='same')(c7)
  u8 = concatenate([u8, c2])
  c8 = Conv2D(32, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(u8)
  c8 = Conv2D(32, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(c8)

  u9 = Conv2DTranspose(16, (2, 2),
                       strides=(2, 2),
                       padding='same')(c8)
  u9 = concatenate([u9, c1], axis=3)
  c9 = Conv2D(16, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(u9)
  c9 = Conv2D(16, (3, 3),
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.L2(1e-7),
              padding='same')(c9)

  outputs = Conv2D(1, (1, 1), activation='relu')(c9)

  return tf.keras.Model(inputs=[inputs], outputs=[outputs])