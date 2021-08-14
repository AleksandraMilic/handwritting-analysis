from recurent_nn import *
from tensorflow import keras
# from tensorflow.keras import layers
import time
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

def find_min_t(X_test):
    min_t = 10**6
    for handwritting in X_test:
        print(len(handwritting))
        if len(handwritting) < min_t:
            min_t = len(handwritting)

    return min_t

def export_results2(score,acc):
    
    ExportToFile="rnn-test-time-series-clf"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv"
    folder = 'D:\\handwritten-analysis\\results-test\\'
    
    prediction = pd.DataFrame(list(zip(score, acc)),
               columns =['scores', 'accuracy'])
    # prediction.index = X_test.index # its important for comparison
    prediction.to_csv(folder+ExportToFile)
    print(prediction)
    return


def real_time(model,X_train, Y_train, X_test, Y_test):
    score_list = []
    accuracy_list = []
    epochs = 1
    
    print('----------real_time------------')
    min_t = find_min_t(X_test)
    step =  min_t // 3
    print('min_t',min_t)

    for i in range(1,min_t,step):

        new_test_X = X_test[:]
        for x in range(len(X_test)):
            new_test_X[x] = new_test_X[x][:i]  ###IGNORE 
        X_test1 = tf.ragged.constant(new_test_X)
        X_test1 = X_test1.to_tensor()


        print(X_test1.shape)

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