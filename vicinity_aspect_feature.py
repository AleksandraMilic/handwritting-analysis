import csv
import glob
import pandas as pd
from xi_yi_features import *



def get_vicinity_aspect(x_list, y_list, r):
    '''Returns ratio of heigh (y_max-y_min) and width (x_max-x_min) of bounding box. 
    Boumding box contains {p_[i-r] + ... + p_[i] + ... + p_[i+r]}, center is p_[i]'''
     
    vicinity_list = []
    list_len = len(x_list)

    for i in range(list_len):

        x_i = x_list[i]
        y_i = y_list[i]
        # print(x_i,y_i)

        
        #lower bound
        if r > i:
            x_1_index = 0
            y_1_index = 0
        else:
            x_1_index = i-r
            y_1_index = i-r
        
        #upper bound
        if i+r > list_len-1:
            x_2_index = list_len-1
            y_2_index = list_len-1
        else:
            x_2_index = i+r
            y_2_index = i+r
            print(i,r)

        print(x_1_index, y_1_index)
        print(x_2_index, y_2_index)


        ### interval of x and y
        interval_x = x_list[x_1_index : x_2_index+1]
        interval_y = y_list[y_1_index : y_2_index+1]
        print(interval_x, interval_y)


        y_max = max(y_list[x_1_index : x_2_index+1])
        y_min = min(y_list[x_1_index : x_2_index+1])
        
        x_max = max(x_list[x_1_index : x_2_index+1])        
        x_min = min(x_list[x_1_index : x_2_index+1])

        print(x_min,x_max,y_min,y_max)
  
        heigh = y_max-y_min
        width = x_max-x_min 
        vicinity = heigh / width
       
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
    print(x_list)

    vicinity_list = get_vicinity_aspect(x_list, y_list, r)
    # print(vicinity_list)
    print(len(x_list))

    print(len(vicinity_list))
    

    
    

