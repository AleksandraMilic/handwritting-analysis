import csv
import glob
import pandas as pd

from math import sqrt
from xi_yi_features import *

def get_cosine_theta(x_list, y_list, t_list):
    '''returns list of cosines of the angle theta'''
    cos_list = []
    cos_list.append(None) ###
    list_len = len(x_list)

    for i in range(1,list_len-1):
        x_1 = int(x_list[i])
        x_2 = int(x_list[i+1])

        y_1 = int(y_list[i])
        y_2 = int(y_list[i+1])

        t_1 = float(t_list[i])
        t_2 = float(t_list[i+1])

        distance = sqrt((x_2-x_1)**2+(y_2-y_1)**2) #euclidean distance
        delta_x = x_2-x_1
        
        cos = delta_x / distance

        cos_list.append(cos)

    cos_list.append(cos) ### last value in column??

    return cos_list


def get_sine_theta(x_list, y_list, t_list):
    '''returns list of sines of the angle theta'''
    sin_list = []
    sin_list.append(None) ###
    list_len = len(x_list)

    for i in range(1,list_len-1):
        x_1 = int(x_list[i])
        x_2 = int(x_list[i+1])

        y_1 = int(y_list[i])
        y_2 = int(y_list[i+1])

        t_1 = float(t_list[i])
        t_2 = float(t_list[i+1])

        distance = sqrt((x_2-x_1)**2+(y_2-y_1)**2) #euclidean distance
        delta_y = y_2-y_1
        
        sin = delta_y / distance

        sin_list.append(sin)

    sin_list.append(sin) ### last value in column??

    return sin_list


if __name__== "__main__":
    
    # folder_csv = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\*.csv'
    # for csv_file in glob.glob(folder_csv):
        # x_list = get_coordinate_x(csv_file)
        # y_list = get_coordinate_y(csv_file)
        # t_list = get_time(csv_file)

        # vx = get_vx_list(x_list, t_list)
        # vy = get_vy_list(y_list, t_list)
        # V = get_V_list(x_list, y_list, t_list)
    
    csv_file = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\a01-000u-03.csv'
    x_list = get_coordinate_x(csv_file)
    y_list = get_coordinate_y(csv_file)
    t_list = get_time(csv_file)



    cos = get_cosine_theta(x_list, y_list, t_list)
    sin = get_sine_theta(x_list, y_list, t_list)
    print(sin)
    print(len(sin))
    print(len(y_list))