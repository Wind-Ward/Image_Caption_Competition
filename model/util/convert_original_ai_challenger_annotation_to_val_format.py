import json
import hashlib
import sys
import io
import jieba
reload(sys)
sys.setdefaultencoding('utf8')
#origin_file_name="/home/windward/symbolic_yangyi/Game/show-attend-and-tell-tensorflow/data/image/annotations/val_new.json"
#converted_file_name="/home/windward/symbolic_yangyi/PycharmProjects/AdaptiveAttentionCAption/data/convert_validation_format_val_3000.json"



origin_file_name="/home/jasperyang/Desktop/Game/show-attend-and-tell-tensorflow/data/image/annotations/val_anno.json"
converted_file_name="/home/jasperyang/Desktop/Game/PycharmProjects/AdaptiveAttentionCaption/data/jieba_convert_validation_format_val_30000.json"




dataset=json.load(open(origin_file_name,"r"))
val_dict={}

annotations=[]
images=[]
id=1
for item in dataset:
    file_name=item["image_id"].split(".")[0]
    image_hash = int(int(hashlib.sha256(file_name).hexdigest(), 16) % sys.maxint)
    for cap in item["caption"]:
        annotations.append({"image_id":image_hash,"id":id,"caption":" ".join(jieba.cut(cap,cut_all=False))})
        id+=1
        images.append({"file_name":file_name,"id":image_hash})
val_dict["annotations"]=annotations
val_dict["images"]=images
val_dict["info"]={"contributor":"He Zheng","description":"CaptionEval","url":"https://github.com/AIChallenger/AI_Challenger.git"\
                  ,"version":"1","year":2017}
val_dict["licenses"]=[{"url":"https://challenger.ai"}]
val_dict["type"]="captions"

#json.dump(unicode(val_dict,open(converted_file_name,"w"),ensure_ascii=False,sort_keys=True, indent=2, separators=(',', ': ')))
with io.open(converted_file_name, 'w', encoding='utf-8') as fd:
    fd.write(unicode(json.dumps(val_dict,
                                ensure_ascii=False, sort_keys=True, indent=2, separators=(',', ': '))))