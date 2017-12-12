import math
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
import io
import json
import random
import jieba
annotation_file="/home/jasperyang/Desktop/Game/show-attend-and-tell-tensorflow/data/image/annotations/val_anno.json"
write_file="/home/jasperyang/Desktop/Game/PycharmProjects/AdaptiveAttentionCaption/data/t_validation_train_11.json"
data=json.load(open(annotation_file,"r"))
result=[]
for item in data:
    #tokens = jieba.cut(str(coco.anns[id]["caption"]), cut_all=False)
    result.append({"image_id":item["image_id"].split(".")[0],"caption":" ".join(jieba.cut(item["caption"][random.randint(0,4)],cut_all=False))})


#json.dump(result,open(write_file,"w"),ensure_ascii=False, sort_keys=True, indent=2, separators=(',', ': '))
with io.open(write_file, 'w', encoding='utf-8') as fd:
    fd.write(unicode(json.dumps(result,
                                ensure_ascii=True, sort_keys=True, indent=2, separators=(',', ': '))))