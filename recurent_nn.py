import os
import multiprocessing as mp
from multiprocessing import Pool #Process, Manager
from decouple import config
from get_model import *
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import tensorflow as tf
from tensorflow.python import keras as k 
from sklearn.model_selection import KFold
import time
from time_series_clf import *


def encoded_data(df1):
    
    encoded_list = []
    classes = df1['class'].values.tolist()
    
    # one hot encoder in keras
    encoded_list = k.utils.np_utils.to_categorical(classes).tolist()
    df1['class_encoded'] = encoded_list
    df1 = df1.drop(columns='class')


    return df1


def create_columns(name, features_files):

    csv_path = os.path.join(features_files + name + '.csv')
    os.environ['CSV'] = csv_path
    csv_path = os.environ.get('CSV')

    file_2 = pd.read_csv(csv_path)
    df2 = DataFrame(file_2)
    features_list = df2.values.tolist()
    # new_column.append(features_list)
    return features_list
    # features_dict[name] = features_list

    # time.sleep(1)







def get_data(writers_classes_file, features_files):
    '''Returns matrix 12186 x 2 (x_data and y_data). Columns: 11th dimensional feature vectors - x data and classes (1-217) - y_data
    inputs: file1: name of csv file, id of writers and class of writers; 
            filess2: features'''
    
    file_1 = pd.read_csv(writers_classes_file) 
    df1 = DataFrame(file_1)
    new_column = [] ##features
    
    # MULTIPROCESSING
    # manager = Manager()
    # features_dict = manager.dict()
    # jobs = []
    # print('start multiprocessing')

    for name in df1['id_handwritting']:
        # print(name)
        features_list = create_columns(name, features_files)
        new_column.append(features_list)
        # p = Process(target=create_columns, args=(name, features_files,features_dict))
        # jobs.append(p)
        # p.start()
    
    # for proc in jobs:
    #     print('jobs')
    #     proc.join()

    # print('end multiprocessing')


    # new_column = features_dict.values()
    df1['features'] = new_column
    df1 = df1.drop(columns='id_handwritting')
    df1 = df1.drop(columns='id_writer')

    df1 = encoded_data(df1)

    print(df1)
### returns dataframe of two columns:  features (of handwrittings)  and and class_encoded

    return df1
  



########################################################################################################################################
def export_results(score,acc,parameters):

    ExportToFile="rnn test-"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv"
    os.environ['EXPORT_FILE'] = ExportToFile
    ExportToFile = os.environ.get('EXPORT_FILE')

    '''
    folder = 'D:\\handwritten-analysis\\results-test\\' #### environ var
    os.environ['FOLDER'] = folder
    folder = os.environ.get('FOLDER')
    '''
    folder = config('FOLDER')

    
    prediction = pd.DataFrame(list(zip(parameters, acc,  score)),
               columns =['parameters', 'accuracy', 'scores'])
    # prediction.index = X_test.index # its important for comparison
    prediction.to_csv(os.path.join(folder+ExportToFile))
    print(prediction)
    return



def get_training_set(data, trainig_index, test_index):
    
    XY = [data[i] for i in trainig_index] ## [[features...],[0,1,0,0...]]...[[],[]]
    X_train = [i[0] for i in XY]
    Y_train = [i[-1] for i in XY] 

    XY = [data[i] for i in test_index]
    X_test = [i[0] for i in XY]
    Y_test = [i[-1] for i in XY]
    
    
    return X_train, Y_train, X_test, Y_test




def optimize_model(data, train_index, test_index, parameters):
    print('-----',train_index, test_index)
    batch_size = parameters[0]
    epochs = parameters[1]
    lr = parameters[2]  
    activation = parameters[3]
    dropout = parameters[4]

    X_train, Y_train, X_test, Y_test = get_training_set(data, train_index, test_index)

    #RAGGED

    X_train = tf.ragged.constant(X_train)
    Y_train = tf.ragged.constant(Y_train)

    X_test1 = tf.ragged.constant(X_test)
    Y_test = tf.ragged.constant(Y_test)

    model = get_model(X_train,lr ,activation,dropout)
    # p1 = Process(target=get_model, args=(data_list, parameters, return_dict_model))
    # p1.start()
    # p1.join()


    X_train = X_train.to_tensor()
    Y_train = Y_train.to_tensor()


    X_test1 = X_test1.to_tensor()
    Y_test = Y_test.to_tensor()


    print(X_train.shape, Y_train.shape)
    # print(X_train, Y_train)

    model.fit(X_train, Y_train, 
            epochs=epochs, batch_size=batch_size,
            validation_data=(X_test1, Y_test))

    score,acc = model.evaluate(X_test1, Y_test, verbose = 2)
    print("Logloss score: %.2f" % (score))
    print("Validation set Accuracy: %.2f" % (acc))
    print("train_index", train_index)

    # return_dict[key] = np.array([score, acc])
    # return_dict[key] = [score, acc]
    result = [score, acc]

    return result
    # return score, acc




# def get_higest_acc(acc, higest_acc, train_index, test_index):
    
#     if higest_acc < acc:
#         higest_acc = acc
#         best_trainig_index = train_index
#         best_test_index = test_index
#         return higest_acc, best_trainig_index, best_test_index
    
#     else:
#         return higest_acc, train_index, test_index





def fit_dataset(data,parameters):
    n_splits=3
    kf = KFold(n_splits=n_splits, shuffle=True) #5 or 3
    # higest_acc = 0
    score_list = []
    accuracy_list = []
    index_list = [[train_index, test_index] for train_index, test_index in kf.split(data)]
    # print(index_list)

    pool = Pool(mp.cpu_count()-1)
    result_list = []
   
    # key = 1
    for train_index, test_index in index_list:

        # print("%s %s" % (train_index, test_index))
        print('--------------------------')
        # score, acc = optimize_model(data, train_index, test_index, parameters, return_dict, key) # return best_trainig_index, best_test_index
        result_ = pool.apply_async(optimize_model, args=(data, train_index, test_index, parameters))
        result_list.append(result_.get())
    print(result_list)
    pool.close()
    pool.join()
        
        # p = Process(target=optimize_model, args=(data, train_index, test_index, parameters, return_dict, key))
        # jobs.append(p)
        # p.start()

    # for proc in jobs:
    #     proc.join()

    # new_dict = {key: value for key, value in sorted(return_dict.items(), key=lambda item: item[1])}
    # print(new_dict)
    # list_ = new_dict.values()

    # score_list = [i[0] for i in list_]
    # accuracy_list = [i[1] for i in list_]

    score_list = [i[0] for i in result_list]
    accuracy_list = [i[1] for i in result_list]

    dataset_index = accuracy_list.index(max(accuracy_list))
    best_train_index = index_list[dataset_index][0]
    best_test_index = index_list[dataset_index][1]

        # higest_acc, best_trainig_index, best_test_index = get_higest_acc(acc, higest_acc, train_index, test_index)        
        
        # score_list.append(score)
        # accuracy_list.append(acc)

    export_results(score_list, accuracy_list, parameters)
    print(best_train_index)
    X_train, Y_train, X_test, Y_test= get_training_set(data, best_train_index, best_test_index) # get the best training set

    # return_dict['1'] = X_train
    # return_dict['2'] = Y_train
    # return_dict['3'] = X_test
    # return_dict['4'] = Y_test


    return X_train, Y_train, X_test, Y_test






if __name__== "__main__":
    
    ####### TEST ALL !!!!!!!
    # os.environ['WRITERS_CL']='D:\\handwritten-analysis\\writer-identification\\writers_classes.csv'
    # writers_classes_file = os.environ.get('WRITERS_CL')
    # writers_classes_file = config('WRITERS_CL')

    
    #####RUNNING ON CPU
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   #run on cpu
    
    
    ####### TEST: 124 files #######
    '''os.environ['WRITERS_CL_TEST'] = 'D:\\handwritten-analysis\\writer-identification\\classification_test.csv'
    writers_classes_file = os.environ.get('WRITERS_CL_TEST')
    print(writers_classes_file)'''
    writers_classes_file = config('WRITERS_CL_TEST')
    '''
    os.environ['FEATURES_FILES'] = 'D:\\handwritten-analysis\\features_data\\testing\\'       
    features_files = os.environ.get('FEATURES_FILES')
    '''
    features_files = config('FEATURES_FILES')


    # from tensorflow.python.client import device_lib
    # print(device_lib.list_local_devices())
    # if tf.test.gpu_device_name():
    #     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    # else:
    #     print("Please install GPU version of TF")
    

    batch_size=1 #1
    epochs=10
    lr = 0.01  
    activation = 'relu'
    dropout = 0.01
    
    parameters = [batch_size, epochs, lr, activation, dropout]


    data_frame = get_data(writers_classes_file, features_files)
    data_list = data_frame.values.tolist()
    

    X_train, Y_train, X_test, Y_test = fit_dataset(data_list, parameters)
    # p1 = Process(target=fit_dataset, args=(data_list, parameters, return_dict))
    # p1.start()
    # p1.join()

    # real_time(X_train, Y_train, X_test, Y_test, parameters)
    
    # list_set = return_dict.values()
    # X_train = list_set[0]
    # Y_train = list_set[1]
    # X_test = list_set[2]
    # Y_test = list_set[3]

    # p2 = Process(target=real_time, args=(X_train, Y_train, X_test, Y_test, parameters))
    # p2.start()
    # p2.join()