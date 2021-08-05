### write all features:
### for each point (x,y)
### number of points is reduced 

import os
import glob
import shutil
from xi_yi_features import *
from writing_direction_feature import *
from vicinity_aspect_feature import *
from vicinity_curliness_feature import *
from velocity_features import *
from synthetic_pressure_feature import *
from curvature_feature import *

from normalize_values import *



def add_files(prev_path, new_path):
    '''copies all csv obtained files in the new folder'''

    for file_path in glob.glob(prev_path):

        file_name = file_path[78:] #[84:]
        # print(file_path)
        csv_name = file_name[:-4] #### id_form ####
        # print(key) 
        new_path_file = new_path + '\\' + csv_name + '.csv'
        # print(path_img,new_path)
        r=shutil.copyfile(file_path, new_path_file)


    return  


def extract_features(file_path,r):
    '''extract 11 features;
    r is the gap parameter'''
    #### POINTS
    x_list = get_coordinate_x(file_path)
    y_list = get_coordinate_y(file_path)
    # print(len(x_list))

    #### TIME
    time_list = get_time(file_path)


    #### VELOCITIES
    vx_list = get_vx_list(x_list, time_list, r)
    vy_list = get_vy_list(y_list, time_list, r)
    V_list = get_V_list(x_list, y_list, time_list, r) 
    
    # print(len(vx_list))
    # print(len(vy_list))


    ####  WRITER DIRECTION
    cosine_list = get_cosine_theta(x_list, y_list,r)
    sine_list = get_sine_theta(x_list, y_list,r)
    
    ####  CURVATURE
    theta_list = get_theta(cosine_list, sine_list)
    
    ####  vICINITY ASPECT
    vicinity_list = get_vicinity_aspect(x_list, y_list, r)
    
    ####  VICINITY CURLINESS
    vicinity_curliness_list = get_vicinity_curliness(x_list, y_list, r)
    
    #### SYNTHETIC WRITER'S PRESSURE AT EACH POINT
    writer_pressure_list = get_synthetic_pressure(time_list)

    all_features = [vx_list,
                    vy_list,
                    V_list,
                    cosine_list,
                    sine_list, 
                    theta_list,
                    vicinity_list,
                    vicinity_curliness_list,
                    writer_pressure_list]

    ### 9 features + x, y coordinates ###
    return all_features




def add_columns(file_path,r):
    column_names = ['Velocity x', 
                    'Velocity y', 
                    'Velocity', 
                    'Cosine theta (writting direction)', 
                    'Sine theta (writting direction)', 
                    'Theta (curvature)',
                    'Vicinity aspect',
                    'Vicinity curliness',
                    'Synthetic pressure'
                    ]
    all_features = extract_features(file_path,r)

    for i in range(len(column_names)):
        # print(len(all_features[i]))
        df = pd.read_csv(file_path)
        # print(df)
        df[column_names[i]] = all_features[i]
        # print(df)
        df.to_csv(file_path, index=False)
    
    df = pd.read_csv(file_path)
    # print(df)
    df_2 = normalize_values(df)
    df_2.to_csv(file_path, index=False)


    return df



if __name__ == "__main__":
    folder_csv = 'D:\\handwritten-analysis\\features_data\\*.csv'
    r = 10
    # prev_folder_csv = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\*.csv'
    for csv_file in glob.glob(folder_csv):
        df_2 = add_columns(csv_file,r)
        print(csv_file)
    #     x = get_coordinate_x(csv_file)
    #     y = get_coordinate_y(csv_file)
    #     t = get_time(csv_file)
    
    
   
    # prev_path = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\*.csv'
    # new_path = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\New folder\\normalized'
    # add_files(prev_path, new_path)
    # csv_file = 'D:\\handwritten-analysis\\features_data\\a01-000u-06.csv'
    
    # df_2 = normalize_values(df)
    # get_to_csv(df_2,csv_file)
