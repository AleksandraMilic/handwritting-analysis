import csv
import glob
import pandas as pd

from math import sqrt, atan
from writing_direction_feature import *

def get_theta(cosine_list, sine_list):
    '''returns list of cosines of the angle theta'''
    theta_list = []
    theta_list.append(None) ###
    list_len = len(cosine_list)

    for i in range(1,list_len-1):
        sine = sine_list[i]
        cosine = cosine_list[i]
        tan = sine / cosine

        arctan_theta = atan(tan)

        theta_list.append(arctan_theta)
     
    theta_list.append(arctan_theta) ### last value in column??

    return theta_list





if __name__== "__main__":
    
    # folder_csv = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\*.csv'
    # for csv_file in glob.glob(folder_csv):
        # x_list = get_coordinate_x(csv_file)
        # y_list = get_coordinate_y(csv_file)
        # t_list = get_time(csv_file)

        # cosine_list = get_cosine_theta(x_list, y_list)
        # sine_list = get_sine_theta(x_list, y_list)

        # theta_list = get_theta(cosine_list, sine_list)
        # print(cos)
        # print(len(cos))
        # print(len(y_list))
    
    csv_file = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\a01-000u-03.csv'
    x_list = get_coordinate_x(csv_file)
    y_list = get_coordinate_y(csv_file)
    t_list = get_time(csv_file)

    cosine_list = get_cosine_theta(x_list, y_list)
    sine_list = get_sine_theta(x_list, y_list)

    theta_list = get_theta(cosine_list, sine_list)
    print(cosine_list[1])
    print(sine_list[1])

    print(theta_list)
    print(len(theta_list))
    print(len(y_list))