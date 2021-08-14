from get_model import *
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from sklearn.model_selection import KFold
import time
from time_series_clf import *


def encoded_data(df1):
    encoded_list = []

    # for name in df1['class']:
        
    classes = df1['class'].values.tolist()
    # print(classes)
    
    # onehot_encoder = OneHotEncoder(sparse=False)
    # label_encoder = LabelEncoder()
    # integer_encoded = label_encoder.fit_transform(encoded_classes)
    # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    # encoded_list = onehot_encoder.fit_transform(classes)


    # one hot encoder in keras
    encoded_list = keras.utils.to_categorical(classes).tolist()

    # print(type(encoded_list))

    df1['class_encoded'] = encoded_list
    df1 = df1.drop(columns='class')


    return df1





def get_data(writers_classes_file, features_files):
    '''Returns matrix 12186 x 2 (x_data and y_data). Columns: 11th dimensional feature vectors - x data and classes (1-217) - y_data
    inputs: file1: name of csv file, id of writers and class of writers; filess2: features'''

    file_1 = pd.read_csv(writers_classes_file) 
    df1 = DataFrame(file_1)

    new_column = [] ##features

    for name in df1['id_handwritting']:
        print(name)
        csv_path = features_files + name + '.csv'
        file_2 = pd.read_csv(csv_path)
        df2 = DataFrame(file_2)
        features_list = df2.values.tolist()
        new_column.append(features_list)

    df1['features'] = new_column
    df1 = df1.drop(columns='id_handwritting')
    df1 = df1.drop(columns='id_writer')

    df1 = encoded_data(df1)

    print(df1)
### returns dataframe of two columns:  features (of handwrittings)  and and class_encoded

    return df1
  
########################################################################################################################################
def export_results(score,acc,train_index, test_index, parameters):

    ExportToFile="rnn test-"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv"
    folder = 'D:\\handwritten-analysis\\results-test\\'
    
    prediction = pd.DataFrame(list(zip(score, acc, train_index, test_index, parameters)),
               columns =['scores', 'accuracy', 'train_index', 'test_index', 'parameters'])
    # prediction.index = X_test.index # its important for comparison
    prediction.to_csv(folder+ExportToFile)
    print(prediction)
    return





def fit_dataset(data):
    batch_size=1
    epochs=1 
    lr = 0.0001




    kf = KFold(n_splits=2, shuffle=True) #5 or 3
    score_list = []
    accuracy_list = []
    train_index_list = []
    test_index_list = []
    activation = 'relu'
    parameters = [batch_size,epochs,lr,activation,None]
    higest_acc = 0

    for train_index, test_index in kf.split(data):
        print("%s %s" % (train_index, test_index))
        train_index_list.append(train_index)
        test_index_list.append(test_index)

        XY = [data[i] for i in train_index] ## [[features...],[0,1,0,0...]]...[[],[]]
        X_train = [i[0] for i in XY]
        Y_train = [i[-1] for i in XY] 

        XY = [data[i] for i in test_index]
        X_test = [i[0] for i in XY]
        Y_test = [i[-1] for i in XY]

       


        #RAGGED
        
        X_train = tf.ragged.constant(X_train)
        Y_train = tf.ragged.constant(Y_train)

        X_test = tf.ragged.constant(X_test)
        Y_test = tf.ragged.constant(Y_test)
        
        model = get_model(X_train,lr,activation)

        X_train = X_train.to_tensor()
        Y_train = Y_train.to_tensor()
        

        X_test = X_test.to_tensor()
        Y_test = Y_test.to_tensor()
        

        print(X_train.shape, Y_train.shape)
        # print(X_train, Y_train)

        model.fit(X_train, Y_train, 
                    epochs=epochs, batch_size=batch_size,
                    validation_data=(X_test, Y_test))

        score,acc = model.evaluate(X_test, Y_test, verbose = 2)
        print("Logloss score: %.2f" % (score))
        print("Validation set Accuracy: %.2f" % (acc))

        score_list.append(score)
        accuracy_list.append(acc)

        if higest_acc < acc:
            higest_acc = acc
            best_trainig_index = train_index
            best_test_index = test_index
    
    export_results(score_list, accuracy_list, train_index_list, test_index_list, parameters)
        



    # return best_trainig_index, best_test_index
    return model, X_train, Y_train, X_test, Y_test






if __name__== "__main__":
    ####### TEST ALL
    # writers_classes_file = 'D:\\handwritten-analysis\\writer-identification\\writers_classes.csv'
    
    ####### TEST: 124 files #######
    writers_classes_file = 'D:\\handwritten-analysis\\writer-identification\\classification_test.csv'
    features_files = 'D:\\handwritten-analysis\\features_data\\testing\\'
    
    data_frame = get_data(writers_classes_file, features_files)
    data_list = data_frame.values.tolist()
    # print(data_list[1][-1])
    model, X_train, Y_train, X_test, Y_test = fit_dataset(data_list)
    real_time(model,X_train, Y_train, X_test, Y_test)