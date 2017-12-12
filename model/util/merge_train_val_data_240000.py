import json
import sys
reload(sys)
sys.setdefaultencoding('utf8')
train_file="/home/jasperyang/Desktop/Game/show-attend-and-tell-tensorflow/data/image/annotations/train_anno.json"
val_file="/home/jasperyang/Desktop/Game/show-attend-and-tell-tensorflow/data/image/annotations/val_anno.json"
merge_all_file="/home/jasperyang/Desktop/Game/PycharmProjects/AdaptiveAttentionCaption/data/train_val_merge_240000.json"
train_json=json.load(open(train_file,"r"))
val_json=json.load(open(val_file,"r"))
train_json.extend(val_json)
json.dump(train_json,open(merge_all_file,"w"),ensure_ascii=False)