import numpy as np
import tensorflow as tf
from tensorflow.python import keras as k 
from tensorflow.python.keras.optimizers import adam_v2 as o

def get_model(X_train,lr,activation,dropout):
    model = k.Sequential()
    max_seq = X_train.bounding_shape()[-1]
    print(max_seq)

    model.add(k.layers.Input(shape=[None, max_seq], dtype=tf.float32, ragged=True))  #Warning: use InputLayer?
    #, batch_size=99
    
    model.add(k.layers.LSTM(128)) #128
    model.add(k.layers.Dropout(dropout)) #0.2


    model.add(k.layers.Dense(64, activation=activation)) #64 #32,relu
    model.add(k.layers.Dropout(dropout)) #0.2

    model.add(k.layers.Dense(4, activation='softmax')) #='softmax' #class - 4????
    # model.add(layers.Dropout(0.2))

    # opt = tf.keras.optimizers.Adam(lr=lr) #0.001, decay=1e-5
    opt = o.Adam(lr=lr) #0.001, decay=1e-5

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.summary()

    return model