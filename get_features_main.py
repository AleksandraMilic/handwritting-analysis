from delete_same_rows import *
from normalize_values import *
from write_features_csv_test import *
import glob

def main():
    
    r = 3
    # prev_path = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\*.csv'
    # new_path = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\New-folder\\normalized'
    
    # add_files(prev_path, new_path)
    new_path = glob.glob('D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\New-folder\\normalized\\*.csv')
    for csv_file in new_path:
        print(csv_file)
        delete_rows_pandas(csv_file)
        df = add_columns(csv_file,r)
        
        # df_2 = normalize_values(df)
        # get_to_csv(df_2,csv_file)

if __name__ == '__main__':
    main()