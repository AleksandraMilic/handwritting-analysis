import shutil



new_path='D:\\handwritten-analysis\\online-analysis\\task1\\a01-000u-01.tif'
path_img='D:\\handwritten-analysis\\online-analysis\\task1\\lineImages-all\\lineImages\\a01\\a01-000\\a01-000u-01.tif'
r=shutil.copyfile(path_img, new_path)
