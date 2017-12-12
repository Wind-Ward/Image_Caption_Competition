import json
import os
import shutil
json_file="/home/windward/symbolic_yangyi/Game/show-attend-and-tell-tensorflow/data/image/annotations/val_new.json"
transfer_dir="/home/windward/symbolic_yangyi/Game/show-attend-and-tell-tensorflow/data/image/val/"
des_dir="/home/windward/symbolic_yangyi/Game/show-attend-and-tell-tensorflow/data/image/val_3000/"
dataset=json.load(open(json_file,"r"))
i=0
for temp in dataset:
    file_name=temp["image_id"]
    shutil.copy(os.path.join(transfer_dir,file_name),des_dir)
    if i%50==0:
        print(i)
    i+=1