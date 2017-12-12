import os
import shutil
annotation_file="/home/jasperyang/Desktop/Game/show-attend-and-tell-tensorflow/data/image/val"
target_dir="/home/jasperyang/Desktop/Game/show-attend-and-tell-tensorflow/data/image/train_val_merge"
file_list=os.listdir(annotation_file)
i=0
for file in file_list:
    shutil.copy(os.path.join(annotation_file,file),target_dir)
    i+=1
    if i%100==0:
        print(i)


