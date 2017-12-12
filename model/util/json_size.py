import json
annotation_file="/home/windward/symbolic_yangyi/data/annotations/train_test.json"
#annotation_file="/home/jasperyang/Desktop/Game/show-attend-and-tell-tensorflow/data/image/annotations/val_new.json"
#write_annotation_file="/home/windward/symbolic_yangyi/PycharmProjects/AdaptiveAttentionCAption/data/adaptive_attention_format_train_11.json"
dataset = json.load(open(annotation_file, 'r'))
print(len(dataset))
#237000
#3000
