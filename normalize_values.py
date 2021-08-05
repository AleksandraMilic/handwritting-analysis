import pandas as pd


def normalize_values(df):
    
    df_2 = df.copy()
    for column in df_2.columns:
        df_2[column] = (df_2[column] - df_2[column].min()) / (df_2[column].max() - df_2[column].min()) 

    return df_2

def get_to_csv(df,csv_file):
    
    df.to_csv(csv_file,index=False)
    print('get csv')

    return
