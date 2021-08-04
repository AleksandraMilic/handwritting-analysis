import csv
import glob
import pandas as pd

from math import sqrt
from xi_yi_features import *

def get_cosine_theta(x_list, y_list,r):
    '''returns list of cosines of the angle theta'''
    cos_list = []
    cos_list.append(1) ###
    list_len = len(x_list)

    for i in range(1,list_len-1):
        x_1 = int(x_list[i])
        try:
            x_2 = int(x_list[i+r])
        except IndexError:
            x_2 = int(x_list[list_len-1])
    

        y_1 = int(y_list[i])
        try:
            y_2 = int(y_list[i+r])
        except IndexError:
            y_2 = int(y_list[list_len-1])
    

        distance = sqrt((x_2-x_1)**2+(y_2-y_1)**2) #euclidean distance
        delta_x = x_2-x_1
        if distance==0:
            cos = None
        else:    
            cos = delta_x / distance

        cos_list.append(cos)

    cos_list.append(cos) ### last value in column??

    return cos_list


def get_sine_theta(x_list, y_list,r):
    '''returns list of sines of the angle theta'''
    sin_list = []
    sin_list.append(0) ###
    list_len = len(x_list)

    for i in range(1,list_len-1):
        x_1 = int(x_list[i])
        try:
            x_2 = int(x_list[i+r])
        except IndexError:
            x_2 = int(x_list[list_len-1])

        y_1 = int(y_list[i])
        try:
            y_2 = int(y_list[i+r])
        except IndexError:
            y_2 = int(y_list[list_len-1])



        distance = sqrt((x_2-x_1)**2+(y_2-y_1)**2) #euclidean distance
        delta_y = y_2-y_1
        if distance == 0:
            sin = None
        else:
            sin = delta_y / distance

        sin_list.append(sin)
####check len lists !!!!!!
    sin_list.append(sin) ### last value in column??

    return sin_list


if __name__== "__main__":
    # r=1
    # folder_csv = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\*.csv'
    # for csv_file in glob.glob(folder_csv):
        
        # x_list = get_coordinate_x(csv_file)
        # y_list = get_coordinate_y(csv_file)
        # t_list = get_time(csv_file)

        # cos = get_cosine_theta(x_list, y_list, r)
        # sin = get_sine_theta(x_list, y_list, r)

    r=1
    csv_file = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\a01-000u-03.csv'
    x_list = get_coordinate_x(csv_file)
    y_list = get_coordinate_y(csv_file)
    t_list = get_time(csv_file)



    cos = get_cosine_theta(x_list, y_list, r)
    sin = get_sine_theta(x_list, y_list, r)
    print(sin)
    print(len(sin))
    print(len(y_list))