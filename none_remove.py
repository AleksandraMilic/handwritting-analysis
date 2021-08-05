import csv
import pandas as pd
import glob

def get_clean_columns(csv_file):

    file = pd.read_csv(csv_file) 
    df = pd.DataFrame(file)
    df = df.fillna(method='bfill').fillna(method='ffill') 
    df.to_csv(csv_file, index=False)

    return

# def check_none(csv_file):
#     data = pd.read_csv(csv_file)
#     if (data['Theta (curvature)'] == data.isnull()):
#         print(csv_file)

#     return


if __name__=='__main__':
    
    folder_csv = 'D:\\handwritten-analysis\\features_data\\testing\\*.csv' #### last folder!!!!! 
    for csv_file in glob.glob(folder_csv): 
        print(csv_file)
        get_clean_columns(csv_file)
    #     get_clean_columns(csv_file)



