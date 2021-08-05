import pandas as pd
import glob


def remove_time(csv_file):
    file = pd.read_csv(csv_file) 
    df = pd.DataFrame(file) 
    # print(df.count())
    # df_1 = df.drop(df['time']<=df['time'].shift(-1)) 
    df_1 = df.drop(columns='time')
    df_1 = df_1.drop(columns='Unnamed: 0')

    # print(df_1.count())
    # print(df_1[5:])
    df_1.to_csv(csv_file,index=False)

    return
 

if __name__ == "__main__":
    folder_csv = 'D:\\handwritten-analysis\\features_data\\testing\\*.csv'
    for csv_file in glob.glob(folder_csv):
        remove_time(csv_file)