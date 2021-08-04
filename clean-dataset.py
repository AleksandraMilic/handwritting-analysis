import glob
import os
### 12186 files
def get_img_names(dataset_img):
    '''returns list of names and paths of files'''
    img_names_list = []
    img_path_list = []

    for writer in glob.glob(dataset_img):
        for img in glob.glob(writer+'\\*'):
            img_name = img[42:]
            img_name = img_name[:-4]
            img_names_list.append(img_name)
            img_path_list.append(img)
            # print(img)


    return img_names_list, img_path_list 

def get_csv_names(folder_csv):
    '''returns  list of names and paths of files'''
    csv_names_list = []
    csv_paths_list = []

    for filename in glob.glob(folder_csv):    
        csv_name = filename[78:]
        csv_name = csv_name[:-4]
        csv_names_list.append(csv_name)
        csv_paths_list.append(filename)

    return csv_names_list, csv_paths_list 



def trash_files_folder_csv(dataset_img, folder_csv):
    '''returns files from files which is not in dataset'''

    trash_list = []
    img_names_list,img_path_list = get_img_names(dataset_img)
    # print(img_names_list)
    # print(len(img_names_list))
    csv_names_list, csv_paths_list = get_csv_names(folder_csv)

    for csv_name, path in zip(csv_names_list, csv_paths_list) :
        if csv_name not in img_names_list:
            trash_list.append(csv_name)
            # print(path)
################## DELETE FILE ############################
            # os.remove(path)


    return trash_list



def trash_files_dataset_img(dataset_img, folder_csv):
    '''returns files from dataset which is not in csv folder'''
    
    trash_list = []
    img_names_list, img_path_list = get_img_names(dataset_img)
    # print(img_names_list)
    # print(len(img_names_list))
    csv_names_list, csv_paths_list = get_csv_names(folder_csv)

    for img_name,path in zip(img_names_list, img_path_list):
        if img_name not in csv_names_list:
            trash_list.append(img_name)
            # print(path)
################## DELETE FILE ############################
            # os.remove(path)



    return trash_list






if __name__== "__main__":
    dataset_img = 'D:\\handwritten-analysis\\new_dataset\\*'
    folder_csv = 'D:\\handwritten-analysis\\online-analysis\\task1\\lineStrokes-all\\lineStrokes-csv\\*.csv'

    # img_number = 0
    # for writer in glob.glob(dataset_img):
    #     for img in glob.glob(writer+'\\*'):
    #         img_number+=1


    trash_list=trash_files_dataset_img(dataset_img, folder_csv)
    print(len(trash_list))

    trash_list2=trash_files_folder_csv(dataset_img, folder_csv)
    print(len(trash_list2))

    print(img_number)
    print(len(glob.glob(folder_csv)))
    trash_files_number = img_number - len(glob.glob(folder_csv))
    print(trash_files_number)