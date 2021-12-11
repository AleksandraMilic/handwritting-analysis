import os
# import multiprocessing as mp
from multiprocessing import Pool, Manager
# from decouple import config
from get_model import *
# import numpy as np
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
    classes = list(map(lambda x: x-1, classes))

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
    df2 = df2.drop(['Sine theta (writting direction)','Theta (curvature)'], axis = 1)

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

    # print(df1)
### returns dataframe of two columns:  features (of handwrittings)  and and class_encoded

    return df1
  



########################################################################################################################################
def export_results(score,acc,parameters):

    ExportToFile="rnn test-"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv"
    os.environ['EXPORT_FILE'] = ExportToFile
    ExportToFile = os.environ.get('EXPORT_FILE')

    
    folder = 'D:/handwritten-analysis/results-test/' #### environ var
    os.environ['FOLDER'] = folder
    folder = os.environ.get('FOLDER')
    print('path')
    # folder = config('FOLDER')

    
    prediction = pd.DataFrame(list(zip(parameters, acc,  score)),
               columns =['parameters', 'accuracy', 'scores'])
    # prediction.index = X_test.index # its important for comparison
    prediction.to_csv(os.path.join(folder+ExportToFile))
    print('saved')
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


# return_list = []
# def optimize_model(data, train_index, test_index, parameters, key, return_dict):
# def get_weights():
#     # os.environ['CL']='classes2.csv'
#     file = os.environ.get('WRITERS_CL') #os.environ.get('CL')
    
#     file1 = pd.read_csv(file)
#     file1 = DataFrame(file1)
#     c = file1['class'].values.tolist()
#     col = list(map(lambda x: x-2, c))
#     # print('-path')
#     # print(col)
#     # print(type(col[0]))
#     l = [0]*216
#     keys = [0]*216
#     for i in range(0,216):
#         class_num = col.count(i)
#         l[i] = class_num
#         keys[i]=i
    
#     zipped = zip(keys,l)
#     d = dict(zipped)
    
#     return d
    
def optimize_model(data, train_index, test_index, parameters, key):
    

    # print('-----',train_index, test_index)
    batch_size = parameters[0]
    epochs = parameters[1]
    lr = parameters[2]  
    activation = parameters[3]
    dropout = parameters[4]

    
    # lock.acquire()

    X_train, Y_train, X_test, Y_test = get_training_set(data, train_index, test_index)

    #RAGGED

    X_train = tf.ragged.constant(X_train)
    Y_train = tf.ragged.constant(Y_train)

    X_test1 = tf.ragged.constant(X_test)
    Y_test = tf.ragged.constant(Y_test)
    print('get model')
    model = get_model(X_train,lr ,activation,dropout, epochs)
    # p1 = Process(target=get_model, args=(data_list, parameters, return_dict_model))
    # p1.start()
    # p1.join()


    X_train = X_train.to_tensor()
    Y_train = Y_train.to_tensor()


    X_test1 = X_test1.to_tensor()
    Y_test = Y_test.to_tensor()

    # print(X_train.shape, X_test1.shape)


    print(X_train.shape, Y_train.shape)
    
    print(X_test1.shape, Y_test.shape)
    # print(X_train, Y_train)
    # t = X_test1[:]
    # t2 = Y_test[:]
    # t3 = X_train[:]
    # t4 = Y_train[:]
    # print('Y_test',Y_test, 't2',t2)
   
    # classWeight = get_weights()
    print('----fit')
    model.fit(X_train, Y_train, 
            epochs=epochs, batch_size=batch_size, #steps_per_epoch=9749//32,
            validation_data=(X_test1, Y_test), verbose=2) #, class_weight=classWeight)
    # model.fit_generator(dataAugmentaion.flow(X_train, Y_train, 
    #     epochs=epochs, batch_size=batch_size,
    #     validation_data=(X_test1, Y_test)))
    print('end')
    # print(X_train, Y_train)
    # print(X_test1, Y_test)
    # print(epochs,batch_size)
    # print('----evaluate')
    score,acc = model.evaluate(X_test1, Y_test, verbose = 0)
    # equality = tf.math.reduce_all(tf.equal(t2, Y_test))
    # equality = tf.math.reduce_all(tf.equal(t2, Y_test))
    # equality2 = tf.math.reduce_all(tf.equal(t, X_test1))
    # equality3 = tf.math.reduce_all(tf.equal(t4, Y_train))
    # equality4 = tf.math.reduce_all(tf.equal(t3, X_train))
    # equality5 = tf.math.reduce_all(tf.equal(X_test1, X_train))

    
    # print("================equality",equality ) 
    # print("================equality",equality2 ) 
    # print("================equality",equality3 ) 
    # print("================equality",equality4 ) 
    # # print("================equality",equality5 ) 



    # print(X_test1, Y_test)

    print("Logloss score: %.2f" % (score))
    print("Validation set Accuracy: %.2f" % (acc))
    print("train_index", train_index)
    
    # return_dict[key] = [score, acc]
        
    

    # return 
    return [key, score, acc]


list_ = []
def f(result):
    list_.append(result)



def fit_dataset(data,parameters):

    
    n_splits=5
    kf = KFold(n_splits=n_splits, shuffle=True) #5 or 3
    # higest_acc = 0
    score_list = []
    accuracy_list = []
    index_list = [[train_index, test_index] for train_index, test_index in kf.split(data)]
    # manager = Manager()
    # return_dict = manager.dict()
    
    # t = [  3,   9,  10,  12,  17,  19,  20,  21,  22,  23,  24,  27,  28,        
    # 32,  35,  36,  37,  38,  42,  43,  44,  46,  47,  48,  51,  52,        
    # 53,  55,  56,  57,  60,  62,  63,  66,  69,  70,  71,  73,  77,        
    # 79,  80,  81,  84,  90,  91,  94,  95,  96,  99, 100, 103, 105,       
    # 106, 107, 108, 109, 114, 115, 117, 119, 120, 123]

    # t2 = [  0,   1,   2,   4,   5,   6,   7,   8,  11,  13,  14,  15,  16,        
    # 18,  25,  26,  29,  30,  31,  33,  34,  39,  40,  41,  45,  49,        
    # 50,  54,  58,  59,  61,  64,  65,  67,  68,  72,  74,  75,  76,        
    # 78,  82,  83,  85,  86,  87,  88,  89,  92,  93,  97,  98, 101,       
    # 102, 104, 110, 111, 112, 113, 116, 118, 121, 122]

    # t = [  0,  1,   2,   3,   5,   6,   7,   8,   9,  12,  13,  15,  16,  17,  18,  19,  23,  25,
#   26,  27,  30,  31,  32,  34,  35,  36,  37,  38,  39,  41,  42,  45,  46,  47,  48,  51,
#   55,  59,  70,  74,  76,  84,  85,  90,  92,  94,  97,  98,  99 ,100, 102 ,105 ,106 ,107,
#  110, 111, 113, 114, 115, 116, 117, 123] 
    # t2 = [  4,  10,  11,  14,  20,  21,  22,  24,  28,  29,  33,  40,  43,  44,  49,  50,  52,  53,
#   54,  56,  57,  58,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  71,  72,  73,  75,
#   77,  78,  79,  80,  81,  82,  83,  86,  87,  88,  89,  91,  93,  95,  96, 101, 103, 104,
#  108 ,109, 112, 118, 119, 120, 121, 122]
    
    '''
    processes = [Process(target=optimize_model, args=(data, train_index, test_index, parameters, key, return_dict)) for (train_index, test_index), key in zip(index_list, range(n_splits))]
    for process in processes:
        process.start()
    
    for process in processes:
        process.join()
    print(return_dict)
    return
    '''

    '''ne radi
    l = [] #single threading version
    for (train_index, test_index), key in zip(index_list, range(n_splits)):
        r = optimize_model(data, test_index, t2, parameters, key)
        l.append(r)
    print(l)
    # '''  
    # ''' 
    pool = Pool(processes=os.cpu_count())
    print(os.cpu_count())
    for (train_index, test_index), key in zip(index_list, range(n_splits)):
        print(1)
        pool.apply_async(optimize_model, args=(data, train_index, test_index, parameters, key,), callback=f) #, key, , callback=f
    for i in range(n_splits):
        print(2)
        pool.close()
    for i in range(n_splits):
        print(3)
        pool.join()
    # print(list_)
    
    # '''
    
    # new_dict = {k: value for  k, value in sorted(return_dict, key=lambda item: k)}
    # l=sorted(list_, key=lambda x: x[0])
    # print(l)
    # new_list  = [[x[1],x[2]] for  x in l]
    new_list = [[x[1],x[2]] for x in sorted(list_, key=lambda x: x[0])]
    print(new_list)
    # new_list = return_dict.values()

    score_list = [i[0] for i in new_list]
    accuracy_list = [i[1] for i in new_list]

 
    dataset_index = accuracy_list.index(max(accuracy_list))
    best_train_index = index_list[dataset_index][0]
    best_test_index = index_list[dataset_index][1]

  

    export_results(score_list, accuracy_list, parameters)
    # print(best_train_index)
    X_train, Y_train, X_test, Y_test= get_training_set(data, best_train_index, best_test_index) # get the best training set


    return X_train, Y_train, X_test, Y_test






if __name__== "__main__":
    
    ####### TEST ALL !!!!!!!
    os.environ['WRITERS_CL']='writers_classes.csv'
    writers_classes_file = os.environ.get('WRITERS_CL')
    
    ####### TEST PART !!!!!!!
    # os.environ['WRITERS_CL']='classes2.csv'
    # writers_classes_file = os.environ.get('WRITERS_CL')

    start = time.time()
    #####RUNNING ON CPU
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   #run on cpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    
    ####### TEST: 124 files #######
    # os.environ['WRITERS_CL_TEST'] = 'classification_test.csv'
    # writers_classes_file = os.environ.get('WRITERS_CL_TEST')
    # print(writers_classes_file)
    
    # os.environ['FEATURES_FILES'] = 'D:\\'       
    # features_files = os.environ.get('FEATURES_FILES')
    # features_files = 'D:/'
    features_files = 'data/'

    

    # from tensorflow.python.client import device_lib
    # print(device_lib.list_local_devices())
    # if tf.test.gpu_device_name():
    #     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    # else:
    #     print("Please install GPU version of TF")
    

    batch_size=2#1
    epochs=200
    lr = 0.1 #0.01
    activation = 'tanh'
    dropout = 0.2
    
    parameters = [batch_size, epochs, lr, activation, dropout]
    # lock = Lock()



    data_frame = get_data(writers_classes_file, features_files)
    data_list = data_frame.values.tolist()
    
    # fit_dataset(data_list, parameters, lock)

    X_train, Y_train, X_test, Y_test = fit_dataset(data_list, parameters)    
    
    # real_time(X_train, Y_train, X_test, Y_test, parameters)
    
    end = time.time()
    print(end - start)
