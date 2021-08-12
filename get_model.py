import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_model(X_train):
    model = keras.Sequential()

    # model.add(layers.Embedding(input_dim=1000, output_dim=4))
    max_seq = X_train.bounding_shape()[-1]
    print(max_seq)

    model.add(layers.Input(shape=[None, max_seq], dtype=tf.float32, ragged=True))  #Warning: use InputLayer?
    #, batch_size=99
    
    model.add(layers.LSTM(128))
    model.add(layers.Dropout(0.2))

    # model.add(layers.LSTM(128, activation='relu'))
    # model.add(layers.Dropout(0.2))

    # model.add(layers.Dense(32, activation='relu'))
    # model.add(layers.Dropout(0.2))

    model.add(layers.Dense(4, activation='softmax')) #class
    # model.add(layers.Dropout(0.2))

    

    # model.compile(
    # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # optimizer="sgd",
    # metrics=["accuracy"],)

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.summary()

    return model