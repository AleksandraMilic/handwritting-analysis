import csv
import glob
import pandas as pd
from xi_yi_features import *
from vicinity_aspect_feature import *
from math import sqrt



def get_vicinity_curliness(x_list, y_list, r):
    '''Returns ratio of heigh (y_max-y_min) and width (x_max-x_min) of bounding box. 
    Boumding box contains {p_[i-r] + ... + p_[i] + ... + p_[i+r]}, center is p_[i]'''
     
    vicinity_list = []
    list_len = len(x_list)

    for i in range(list_len):
  
        heigh, width = get_dimension_box(x_list, y_list, r, list_len, i)
        
        if width == 0 or heigh == 0:
            vicinity = None
        else:
            diagonal = sqrt(heigh**2 + width**2)    
            vicinity = diagonal / min(heigh, width)
       
        vicinity_list.append(vicinity)


    return vicinity_list


if __name__ == "__main__":
    # r=1
    # folder_csv = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\*.csv'
    # for csv_file in glob.glob(folder_csv):

        # x_list = get_coordinate_x(csv_file)
        # y_list = get_coordinate_y(csv_file)
        # t_list = get_time(csv_file)

        # cosine_list = get_cosine_theta(x_list, y_list,r)
        # sine_list = get_sine_theta(x_list, y_list,r)

        # theta_list = get_theta(cosine_list, sine_list)
        # print(cos)
        # print(len(cos))
        # print(len(y_list))
    
    r = 2
    csv_file = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\a01-000u-03.csv'
    x_list = get_coordinate_x(csv_file)

    y_list = get_coordinate_y(csv_file)
    # print(x_list)

    vicinity_list = get_vicinity_curliness(x_list, y_list, r)
    print(vicinity_list)
    print(len(x_list))

    print(len(vicinity_list))
    