import pandas as pd
import glob




def create_dict(form_txt):
    '''key: id_handwritting; value: names of files
       key: id_writer; value:  id of writers'''
    
    writer_list = glob.glob(dataset_path+"*") #list of created folders in dataset

    writers_id_list = []
    forms_id_list = []


    for writer_path in writer_list:
        
        writer_id = writer_path[36:] 
        writer_path = writer_path +'\\*'
        # print(writer_path)

        for csv_file in glob.glob(writer_path):
            # print(writer_path)
            
            form_id = csv_file[42:]
            form_id = form_id[:-4]

            forms_id_list.append(form_id)
            writers_id_list.append(writer_id)

    
    # print(len(forms_id_list),len(writers_id_list))
    # print(forms_id_list, writers_id_list)
    dict_ = {'id_handwritting' : forms_id_list,
            'id_writer' : writers_id_list}
    
    
    return dict_, writers_id_list






def generate_list_classes(writers_id_list):
    
    all_classes = []
    class_id = 1
    all_classes.append(class_id)
    for i in range(1,len(writers_id_list)): ### 217 writers 
        if writers_id_list[i] == writers_id_list[i-1]:
            all_classes.append(class_id)
        else: 
            class_id += 1 
            all_classes.append(class_id)
    
    return all_classes



def add_class_column(df,writers_id_list):
           
    df['class'] = generate_list_classes(writers_id_list)
    # print(df)

    return df




###########
def create_file_classes(path_file,  dataset_path):
    
    dict_, writers_id_list = create_dict( dataset_path)
    # print(dict_)
    df = pd.DataFrame(dict_)
    # print(df)

    df = add_class_column(df, writers_id_list)
    # print(df)

##### SAVE FILE #####
    path_file = path_file + '\\' + '.csv'
    df.to_csv(path_file, index=False)

    return

###########




if __name__ == "__main__":    

    dataset_path = 'D:\\handwritten-analysis\\new_dataset\\*'
    path_file = 'D:\\handwritten-analysis\\writer-identification'

    create_file_classes(path_file,  dataset_path)

    