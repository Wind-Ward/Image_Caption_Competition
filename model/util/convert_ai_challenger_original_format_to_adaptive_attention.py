import json
import random
import argparse
import sys
reload(sys)
sys.setdefaultencoding('utf8')
annotation_file="/Users/yinfeng/PycharmProjects/AdaptiveAttentionCaption/data/train_val_merge_240000.json"
write_annotation_file="/Users/yinfeng/PycharmProjects/AdaptiveAttentionCaption/data/adaptive_attention_format_train_val_240000.json"
dataset = json.load(open(annotation_file, 'r'))
new_dataset=[]
ids=0
for item in dataset:
    for cap in item["caption"]:
        temp = {}
        temp["image_id"]=item["image_id"]
        temp["caption_id"]=ids
        ids+=1
        temp["caption"]=cap
        new_dataset.append(temp)
random.shuffle(new_dataset)
json.dump(new_dataset,open(write_annotation_file,"w"),ensure_ascii=False)
print(ids)


#1185000
#15000