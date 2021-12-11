import numpy as np
import tensorflow as tf
from tensorflow.python import keras as k 
from tensorflow.python.keras.optimizers import adam_v2 as o
# from keras import losses
# from tensorflow.python.keras.optimizers import SGD

def get_model(X_train,lr,activation,dropout,epochs):
    # decay = lr/epochs 

    model = k.Sequential()
    max_seq = X_train.bounding_shape()[-1]
    print('max_seq',max_seq)
    
    # model.add(k.layers.LSTM(32, input_shape=[32,None, 11], dtype=tf.float32)) #, return_sequence=True
    model.add(k.layers.Input(shape=[None, max_seq], dtype=tf.float32, ragged=True))  #Warning: use InputLayer?

    #, batch_size=99
    # model.add(k.layers.Dropout(dropout))
    
    model.add(k.layers.LSTM(64)) #128
    model.add(k.layers.Dropout(dropout)) #0.2

    model.add(k.layers.Dense(64, activation=activation)) #64 #32,relu
    model.add(k.layers.Dropout(0.2))
    
    # model.add(k.layers.Dense(64, activation=activation)) #64 #32,relu
    # model.add(k.layers.Dropout(0.2))

    # model.add(k.layers.Dense(32, activation=activation)) #64 #32,relu
    # model.add(k.layers.Dropout(0.2))

    # model.add(k.layers.Dense(32, activation=activation)) #64 #32,relu
    # model.add(k.layers.Dropout(dropout))

    model.add(k.layers.Dense (217, activation='softmax')) #='softmax' #class - 218????
    # model.add(layers.Dropout(0.2))

    # opt = tf.keras.optimizers.Adam(lr=lr) #0.001, decay=1e-5
    # opt = tf.keras.optimizers.Nadam(lr=lr, decay=1e-5)
    
    # lr_schedule = k.optimizers.schedules.ExponentialDecay(
    # initial_learning_rate=1e-2,
    # decay_steps=10000,
    # decay_rate=0.9)
    
    opt = o.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5) #0.001, decay=1e-5
    # opt = o.Adagrad(lr=lr, decay=decay) #0.001, decay=1e-5

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    # model.compile(loss=k.losses.mean_squared_error, optimizer=opt, metrics=["accuracy"])
    model.summary()

    return model
