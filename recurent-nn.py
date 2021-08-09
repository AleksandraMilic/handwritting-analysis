import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from tensorflow.keras.models import Sequential
from tensorflow import keras
from sklearn.model_selection import KFold


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


def fit_dataset(data):

    kf = KFold(n_splits=5)

    for train_index, test_index in kf.split(data):
        print("%s %s" % (train_index, test_index))

        X_Y = [data[i] for i in train_index] ## [[features...],[0,1,0,0...]]...[[],[]]
        X_train = [i[:-1] for i in X_Y]
        Y_train = [i[-1] for i in X_Y]

        X_Y = [data[i] for i in test_index]
        X_test = [i[:-1] for i in X_Y]
        Y_test = [i[-1] for i in X_Y]

        print(X_train, Y_train)
        print(X_test, Y_test)
        
        break
    # train = dataset[train_index]
    # test = dataset[test_index]
    # print(train, test)






    return






if __name__== "__main__":
    # writers_classes_file = 'D:\\handwritten-analysis\\writer-identification\\writers_classes.csv'
    
    ####### TEST: 124 files #######
    writers_classes_file = 'D:\\handwritten-analysis\\writer-identification\\classification_test.csv'
    features_files = 'D:\\handwritten-analysis\\features_data\\testing\\'
    
    data_frame = get_data(writers_classes_file, features_files)
    data_list = data_frame.values.tolist()
    # print(data_list[1][-1])
    fit_dataset(data_list)