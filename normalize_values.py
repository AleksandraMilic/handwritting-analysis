import pandas as pd


def normalize_values(df):

    df_2 = (df - df.min()) / (df.max() - df.min())
    print(normalize_values)

    return df_2

def get_to_csv(df,csv_file):
    
    df.to_csv(csv_file,index=False)
    print('get csv')

    return
