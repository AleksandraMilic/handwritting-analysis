from recurent_nn import *
from get_model import *
import tensorflow as tf 
from tensorflow import keras
# from decouple import config
# from tensorflow.keras import layers
import time
import numpy as np
import pandas as pd
import os
from pandas.core.frame import DataFrame

def find_min_t(X_test):
    min_t = 10**6
    for handwritting in X_test:
        # print(len(handwritting))
        if len(handwritting) < min_t:
            min_t = len(handwritting)

    return min_t

def export_results2(score,acc):
    
    ExportToFile="rnn-test-time-series-clf"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv"
    os.environ['EXPORT'] = ExportToFile
    ExportToFile = os.environ.get('EXPORT')
    folder = 'D:/handwritten-analysis/results-test/'
    os.environ['FOLDER'] = folder
    folder = os.environ['FOLDER']
    
    prediction = pd.DataFrame(list(zip(score, acc)),
               columns =['scores', 'accuracy'])
    # prediction.index = X_test.index # its important for comparison
    prediction.to_csv(os.path.join(folder+ExportToFile))
    print(prediction)
    return


def real_time(X_train, Y_train, X_test, Y_test,parameters):
    score_list = []
    accuracy_list = []
    epochs = parameters[1]
    lr = parameters[2]  
    activation = parameters[3]
    dropout = parameters[4]

    X_train = tf.ragged.constant(X_train)
    Y_train = tf.ragged.constant(Y_train)
    Y_train = Y_train.to_tensor()
    Y_test = tf.ragged.constant(Y_test).to_tensor()

    
    model = get_model(X_train,lr,activation,dropout,epochs)

    X_train = X_train.to_tensor()

    
    print('----------real_time------------')
    min_t = find_min_t(X_test)
    step =  min_t // 10 #only min_t
    print('min_t',min_t)

    for i in range(1,min_t+1,step):

        new_test_X = X_test[:]
        for x in range(len(X_test)):
            new_test_X[x] = new_test_X[x][:i]  ###cut data 
            # if x==0:
                # print('x',new_test_X[x])

        X_test1 = tf.ragged.constant(new_test_X).to_tensor()
    


        print('-----X_test1.shape',X_test1.shape)

        model.fit(X_train, Y_train, 
                    epochs=epochs, batch_size=1,
                    validation_data=(X_test1, Y_test))

        score,acc = model.evaluate(X_test1, Y_test, verbose = 2)
        print("Logloss score: %.2f" % (score))
        print("Validation set Accuracy: %.2f" % (acc))

        score_list.append(score)
        accuracy_list.append(acc)

    export_results2(score_list, accuracy_list)




    return