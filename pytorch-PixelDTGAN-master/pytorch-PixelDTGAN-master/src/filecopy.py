import os
import shutil
src_dir = 'E:/work/ML/infinity/lookbook/data/'
dest_dir = 'C:/Users/sravyay/Pictures/createdData/data/'
files = os.listdir(src_dir)
curr = 'null'
prev = 'null'
#counter = 0
for file in files:
    # print(src_dir+file)
    # print(dest_dir+file)
    # break
    if file.endswith('.jpg'):
        print(file)
        curr = (file.split('_')[0])
        if(prev != curr):
            os.mkdir(dest_dir+curr)
            shutil.copy(src_dir + file, dest_dir + curr + '/' + file)
        else:
            shutil.copy(src_dir + file, dest_dir + curr + '/' + file)
        prev = curr
    # counter += 1
    # if(counter == 20):
    #     break
    #     #shutil.copy("Full path to file", "Full path to dest folder")