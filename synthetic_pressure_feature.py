import numpy as np
import csv
import glob
import pandas as pd
from xi_yi_features import *


def get_base_pressure(m,s):
    '''returns base pressure'''
    base_pressure = np.random.normal(m, s)

    return base_pressure

def get_epsilon(delta_p):
    '''returns epsilon;
    parameters are m=0 and delta_p'''

    epsilon = np.random.normal(0, s)

    return epsilon


def get_synthetic_pressure(m,s,delta_p,time_list):
    '''returns synthetic pressures of one handwriting created using sampling by normal distribution'''
    writer_pressure_list = []
    len_time_list = len(time_list)
    
    base_pressure = get_base_pressure(m,s)
    writer_pressure_list.append(base_pressure)

    for i in range(1,len_time_list):
        prev_pressure = writer_pressure_list[i-1]
        epsilon = get_epsilon(delta_p)
        pressure = prev_pressure + epsilon

        writer_pressure_list.append(pressure)

    return writer_pressure_list


if __name__== "__main__":
    m = 500
    s = 10
    delta_p = 10
    # folder_csv = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\*.csv'
    # for csv_file in glob.glob(folder_csv):
        # time = get_time(csv_file)
        # writer_pressure_list = get_synthetic_pressure(m,s,delta_p,time_list)


    csv_file = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\a01-000u-03.csv'
    time_list = get_time(csv_file)
    print(len(time_list))

    writer_pressure_list = get_synthetic_pressure(m,s,delta_p,time_list)
    print(writer_pressure_list)
    print(len(writer_pressure_list))
