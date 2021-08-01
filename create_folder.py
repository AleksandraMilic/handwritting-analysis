import os
import glob
import shutil



def convert_txt(form_txt):
    '''convert 2 columns from form.txt to list'''
    
    with open(form_txt) as f:
        forms_list = [[(line.rstrip()).split()[0], (line.rstrip()).split()[1]] for line in f]

    return forms_list




def add_folders(dataset_path, form_txt):
    '''add folder for each writer (class)'''
    forms_list = convert_txt(form_txt)
    id_writers_list = [i[1] for i in forms_list]
    id_writers_list = list(set(id_writers_list))

    for id_writer in id_writers_list:
        writer_folder_path = os.path.join(dataset_path, id_writer)
        ############################################################################################
        os.mkdir(writer_folder_path) 

    return 



def create_dict(dataset_path, form_txt):
    '''key: id_form; value: path of writer's folder'''
    
    # folders_list = glob.glob(dataset_path+"*") #list of created folders in dataset

    forms_list = convert_txt(form_txt)
    id_forms_list = [i[0] for i in forms_list]
    folders_list = [dataset_path+i[1] for i in forms_list]

    dict_ = dict(zip(id_forms_list, folders_list))
    
    return dict_



def add_files(prev_dataset, form_txt, dataset_path):
    dict_ = create_dict(dataset_path, form_txt) 
    # print(dict_)

    for a00 in glob.glob(prev_dataset):
        for a00_000 in glob.glob(a00+"\\*"):
            for path_img in glob.glob(a00_000+"\\*.tif"):
                
                img_name = path_img[84:]
                key = img_name[:-7] #### id_form ####
                # print(key)
                writer_folder_path = dict_[key] #z01-010p?????
                new_path = writer_folder_path +'\\' + img_name 
                # print(path_img,new_path)
                r=shutil.copyfile(path_img, new_path)


    return


            




if __name__ == "__main__":    

    dataset_path = 'D:\\handwritten-analysis\\new_dataset\\'
    form_txt = 'D:\\handwritten-analysis\online-analysis\\task1\\forms.txt'
    prev_dataset = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineImages-all\\lineImages\\*'

    add_folders(dataset_path, form_txt)
    add_files(prev_dataset, form_txt, dataset_path)
    

