import csv
import glob
import pandas as pd
from math import sqrt
from xi_yi_features import *


def get_vx_list(x_list, t_list):
    '''returns list of velosity Vx'''
    vx_list = []
    vx_list.append(0)
    list_len = len(x_list)

    for i in range(1,list_len-1):
        x_1 = int(x_list[i])
        x_2 = int(x_list[i+1])

        t_1 = float(t_list[i])
        t_2 = float(t_list[i+1])

        delta_x = abs(x_2-x_1)
        delta_t = t_2-t_1

        vx = delta_x / delta_t

        vx_list.append(vx)
    vx_list.append(vx) ### last value in column

    return vx_list



def get_vy_list(y_list, t_list):
    '''returns list of velosity Vy'''
    vy_list = []
    vy_list.append(0)
    list_len = len(y_list)

    for i in range(1,list_len-1):
        y_1 = int(y_list[i])
        y_2 = int(y_list[i+1])

        t_1 = float(t_list[i])
        t_2 = float(t_list[i+1])

        delta_y = abs(y_2-y_1)
        delta_t = t_2-t_1

        vy = delta_y / delta_t

        vy_list.append(vy)
    vy_list.append(vy) ### last value in column
    return vy_list



def get_V_list(x_list, y_list, t_list):
    '''returns list of velosity V at point (x,y)'''
    V_list = []
    V_list.append(0)
    list_len = len(x_list)

    for i in range(1,list_len-1):
        x_1 = int(x_list[i])
        x_2 = int(x_list[i+1])

        y_1 = int(y_list[i])
        y_2 = int(y_list[i+1])

        t_1 = float(t_list[i])
        t_2 = float(t_list[i+1])

        distance = sqrt((x_2-x_1)**2+(y_2-y_1)**2) #euclidean distance
        delta_t = t_2-t_1

        
        V = distance / delta_t

        V_list.append(V)

    V_list.append(V) ### last value in column

    return V_list

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



    vx = get_vx_list(x_list, t_list)
    vy = get_vy_list(y_list, t_list)
    V = get_V_list(x_list, y_list, t_list)    
    print(V)
    print(len(V))
    print(len(y_list))
    