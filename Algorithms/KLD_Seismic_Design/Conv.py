import tensorflow as tf
from tensorflow.keras.layers import *

def conv_model(DataTrain):
    input_size = DataTrain.shape[1:]
    inputs = tf.keras.Input(shape=input_size)

    x = Conv2D(128, 3, padding='same', activation="relu")(inputs)
    x = Conv2D(128, 3, padding='same', activation="relu")(x)
    x = Conv2D(128, 3, padding='same', activation="relu")(x)
    x = Conv2D(128, 3, padding='same', activation="relu")(x)
    outputs = Conv2D(input_size[-1], 3, padding='same', activation="sigmoid")(x)
    return tf.keras.Model(inputs=[inputs], outputs=[outputs])