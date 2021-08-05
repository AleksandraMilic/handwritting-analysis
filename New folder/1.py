# import sklearn
import cv2
import glob
import numpy as np
import os.path
from os import path



def huMoments(img):
    im=cv2.imread(img,0)
    moments = cv2.moments(im)
    cv2.imshow('image window', im)
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    huMoments = np.ndarray.tolist(huMoments)
    for i in range(7):
        f.append(huMoments[i][0])
    
    return features
    
    # print("type",type(huMoments))





def shape_features(folder,data,folder_name):
    ''' form_id - handwritting (list of id)
        folder - main folder of binary images (string)
        data - id of img and writers '''
    # features = []
    
    f = [] #list of feature vector with its class (writer)

    for a00 in folder:
        # print(a00) 
        for a00_000 in glob.glob(a00+"\\*"):

            for a00_000x, writer in data:
                if (a00_000 == folder_name + a00_000 + a00_000x[:-1]) or (a00_000 == folder_name + a00_000 + a00_000x):
                    first_img_number = 0
                    second_img_number = 0
                    path_img = a00_000 + "\\" + a00_000x + str(first_img_number) + str(second_img_number) + ".tif"
                    print(path_img) 

                    if path.exists(path_img)==True:
                        path_img = img
                        features = huMoments(img)
                        class_ = writer
                        f.append([features, class_])
                        print(path_img) 

                    
                    second_img_number += 1
                    path_img = a00_000 + "\\" + a00_000x + str(first_img_number) + str(second_img_number) + ".tif"

    # jednocifreni id
                    while path.exists(path_img)==True and second_img_number<=9:
                        img = path_img
                        features = huMoments(img)
                        class_ = writer
                        f.append([features, class_])
                        print(path_img) 


                        second_img_number += 1
                        path_img = a00_000 + "\\" + a00_000x + str(first_img_number) + str(second_img_number) + ".tif"
                        


    # dvocifreni id
                    if second_img_number>9:
                        img_number = 10
                        path_img = a00_000 + "\\" + a00_000x + str(img_number) + ".tif"
                        print(path_img) 


                        while path.exists(path_img)==True:
                            img = path_img
                            features = huMoments(img)
                            class_ = writer
                            f.append([features, class_])
                            print(path_img) 


                            img_number += 1
                            path_img = a00_000 + "\\" + a00_000x + str(img_number) + ".tif"





            # print(1) 

            # hu moments - 7 moments
                # Calculate Moments
       
            # print(huMoments)
            # coefficients


    return f

if __name__ == "__main__":    

    data = [] #id of img and writers
    id_img = []
    ##open forms - image folders and writers
    with open('D:\handwritten-analysis\online-analysis\\task1\\forms.txt') as f:
        data = [[(line.rstrip()).split()[0], (line.rstrip()).split()[1]] for line in f]

    # with open('D:\handwritten-analysis\online-analysis\\task1\\forms.txt') as f:
    #     img_data = [(line.rstrip()).split()[0] for line in f]


    #folder lineImage
    folder_1 = glob.glob('D:\\handwritten-analysis\\online-analysis\\task1\\lineImages-all\\lineImages\\*')
    folder_2 = glob.glob('D:\\handwritten-analysis\\online-analysis\\task1\\lineImages-all\\testlineImages\\*')
    folder_name = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineImages-all\\testlineImages\\'

    features=shape_features(folder_1,data,folder_name)
    print(features)


    
