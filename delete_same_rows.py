from xi_yi_features import *
import pandas as pd
import glob

def delete_rows_pandas(csv_file):
    '''delete rows which has a less or equal value of time than previous row'''

    file = pd.read_csv(csv_file) 
    df = pd.DataFrame(file) 
    print(df.count())
    # df_1 = df.drop(df['time']<=df['time'].shift(-1)) 
    df_1 = df.drop_duplicates(subset=['time'])
    print(df_1.count())
    print(df_1[5:])

    return df_1


def delete_same_rows(csv_file):
    '''delete rows which has a less or equal value of time than previous row'''

    file = pd.read_csv(csv_file)
    file_2 = pd.DataFrame(file) 
    # print(file_2)


    time_list = get_time(csv_file)
    

    drop_rows_list = []
    # print(len(time_list)-1, len(file))

    for i in range(len(time_list)-1):
        if i > 1:

                # print("timelist : %d %f " % (i, time_list[i]))
            # pass
            # print(time_list[i], time_list[i-1])
            if time_list[i] <= time_list[i-1]:
                ## delete row
                drop_rows_list.append(i)
    print(drop_rows_list)
    # print(len(drop_rows_list))
    print(file_2.count())
    df = file_2.drop(drop_rows_list, inplace=True)
    print(file_2)

    # file_2.to_csv(csv_file,index=False)
    # print(file_2.count())
    return df
 




if __name__ == "__main__":
    
    # folder_csv = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\*.csv'

    folder_csv = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\New folder\\*.csv'
    for csv_file in glob.glob(folder_csv):
        print(csv_file)
        delete_rows_pandas(csv_file)
