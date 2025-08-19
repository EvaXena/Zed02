import tensorflow as tf
from tensorflow import keras
from keras import layers
from model.layers.tanhlu import Tanhlu
from tensorflow.keras import layers, initializers, models
def O1000():
    input_shape = (256,256,3)
    model_input = keras.Input(shape=input_shape)

    x = layers.Conv2D(filters=32,kernel_size=3,strides=2,use_bias=False,name = 'conv_1')(model_input)
    x = layers.BatchNormalization(name = 'bn_1')(x)
    x = layers.Activation('tanh',name = 'tanh_1')(x)

    x = layers.Conv2D(filters=64,kernel_size=3,strides=2,padding='same',use_bias=False,name = 'conv_2')(x)
    x = layers.BatchNormalization(name = 'bn_2')(x)
    x = layers.Activation('tanh',name = 'tanh_2')(x)

    x = layers.Conv2D(filters=128,kernel_size=3,strides=1,padding='same',use_bias=False,name = 'conv_3')(x)
    x = layers.BatchNormalization(name = 'bn_3')(x)
    x = layers.Activation('tanh',name = 'tanh_3')(x)
    x = layers.UpSampling2D(size=(2,2),interpolation='nearest',name=f"up_1")(x)
    x = layers.UpSampling2D(size=(2,2),interpolation='nearest',name=f"up_2")(x)

    x =  layers.Conv2D(filters=1,kernel_size=1,strides=1,padding='same',name = 'conv_out')(x)

    model = keras.Model(inputs = model_input,outputs = x,name = 'O1000_tanh')

    model.summary()

    return model