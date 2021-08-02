import csv
import glob
##for each 10th point##

def get_coordinate_x(csv_file):
    '''returns list of x coordinates'''
    file = open(csv_file, "r")
    csv_reader = list(csv.reader(file))
    x_list = []
    len_csv = len(csv_reader)
    
    for i in range(1,len_csv,9):
        x_list.append(csv_reader[i][1])

    
    
    # for each point
    
    # for row in csv_reader:
    #     x_list.append(row[1])
    # x_list.pop(0)

    return(x_list)

def get_coordinate_y(csv_file):
    '''returns list of y  coordinates'''
    file = open(csv_file, "r")
    csv_reader = list(csv.reader(file))
    y_list = []
    len_csv = len(csv_reader)
    
    for i in range(1,len_csv,9):
        y_list.append(csv_reader[i][2])


    # for row in csv_reader:
    #     y_list.append(row[2])
    #     y_list.pop(0)

    return(y_list)

def get_time(csv_file):
    '''returns list of time'''
    file = open(csv_file, "r")
    csv_reader = list(csv.reader(file))
    t_list = []
    len_csv = len(csv_reader)


    for i in range(1,len_csv,9):
        t_list.append(csv_reader[i][3])


    # for row in csv_reader:
    #     t_list.append(row[3])
        # t_list.pop(0)
        
    
    return(t_list)


if __name__== "__main__":
    
    # folder_csv = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\*.csv'
    # for csv_file in glob.glob(folder_csv):
    #     x = get_coordinate_x(csv_file)
    #     y = get_coordinate_y(csv_file)
    #     t = get_time(csv_file)
    
    csv_file = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\a01-000u-03.csv'
    x = get_coordinate_x(csv_file)
    y = get_coordinate_y(csv_file)
    t = get_time(csv_file)

    print(t)