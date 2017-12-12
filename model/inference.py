# -*- coding:utf-8 -*-
import math
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from utils import coco_eval, to_var
from data_loader import get_loader
from adaptive import Encoder2Decoder
from build_vocab_ai_challenger import Vocabulary
from torch.autograd import Variable
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import io


def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable( x, volatile=volatile )

class InferenceLoader(Dataset):
    def __init__(self, img_path, img_transform=None,
                 loader=datasets.folder.default_loader):
        '''
        Customized COCO loader to get Image ids and Image Filenames
        root: path for images
        ann_path: path for the annotation file (e.g., caption_val2014.json)
        '''
        self.root = img_path
        self.file_list = os.listdir(img_path)
        self.img_transform = img_transform
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = self.loader(os.path.join(self.root,img_path))
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img,img_path.split(".")[0]

    def __len__(self):
        return len(self.file_list)


def main(args):
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    adaptive = Encoder2Decoder(args.embed_size, len(vocab), args.hidden_size)
    adaptive.load_state_dict(torch.load(args.pretrained))
    if torch.cuda.is_available():
        adaptive.cuda()
    adaptive.eval()

    transform = transforms.Compose([
        transforms.Scale((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    inference_data_loader = torch.utils.data.DataLoader(
        InferenceLoader(args.image_dir,img_transform=transform),
        batch_size=args.eval_size,
        shuffle=False, num_workers=args.num_workers,
        drop_last=False)
    results = []
    print('---------------------Start Inference on AI-challenger dataset-----------------------')

    for i,(images,file_prefix) in enumerate(inference_data_loader):
        images = to_var(images)
        generated_captions = adaptive.sampler_beam_search(images,args.beam_size)
      
        sampled_caption = []
        #_generated_captions=generated_captions.cpu().data.numpy()
        for word_id in generated_captions:
            #print(word_id.int())
            word = vocab.idx2word[int(word_id.cpu().data.numpy())]
            if word == '<end>':
                break
            else:
                sampled_caption.append(word)

        sentence = ''.join(sampled_caption[1:])
        temp = {'image_id': file_prefix[0], 'caption': sentence}
        results.append(temp)

        # Disp evaluation process
        if (i + 1) % 10 == 0:
            print('[%d/%d]' % ((i + 1), len(inference_data_loader)))

    #json.dump(results,open(args.inference_output_json,"w"),ensure_ascii=False,sort_keys=True, indent=2, separators=(',', ': '))
    with io.open(args.inference_output_json, 'w', encoding='utf-8') as fd:
        fd.write(unicode(json.dumps(results,
                                    ensure_ascii=False, sort_keys=True, indent=2, separators=(',', ': '))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', default='self', help='To make it runnable in jupyter')
    parser.add_argument('--model_path', type=str, default='./models',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str,
                        default='/home/windward/symbolic_yangyi/PycharmProjects/AdaptiveAttentionCAption/data/vocab_ai_challenger_11.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str,
                        default='/home/windward/symbolic_yangyi/PycharmProjects/AdaptiveAttentionCAption/data/image_resize_11/',
                        help='directory for resized training images')
    parser.add_argument('--caption_path', type=str,
                        default='/home/windward/symbolic_yangyi/PycharmProjects/AdaptiveAttentionCAption/data/adaptive_attention_format_train_11.json',
                        help='path for train annotation json file')
    parser.add_argument('--caption_val_path', type=str,
                        default='./data/annotations/karpathy_split_val.json',
                        help='path for validation annotation json file')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing log info')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed for model reproduction')
    parser.add_argument('--inference_output_json', type=str, default="./inference_output.json",
                        help='random seed for model reproduction')

    # ---------------------------Hyper Parameter Setup------------------------------------

    # CNN fine-tuning
    parser.add_argument('--fine_tune_start_layer', type=int, default=5,
                        help='CNN fine-tuning layers from: [0-7]')
    parser.add_argument('--cnn_epoch', type=int, default=20,
                        help='start fine-tuning CNN after')

    # Optimizer Adam parameter
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='alpha in Adam')
    parser.add_argument('--beta', type=float, default=0.999,
                        help='beta in Adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                        help='learning rate for the whole model')
    parser.add_argument('--learning_rate_cnn', type=float, default=1e-4,
                        help='learning rate for fine-tuning CNN')

    # LSTM hyper parameters
    parser.add_argument('--embed_size', type=int, default=768,
                        help='dimension of word embedding vectors, also dimension of v_g')
    parser.add_argument('--hidden_size', type=int, default=1024,
                        help='dimension of lstm hidden states')

    # Training details
    parser.add_argument('--pretrained', type=str, default='', help='start from checkpoint or scratch')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)  # on cluster setup, 60 each x 4 for Huckle server

    # For eval_size > 30, it will cause cuda OOM error on Huckleberry
    parser.add_argument('--eval_size', type=int, default=1)  # on cluster setup, 30 each x 4
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--lr_decay', type=int, default=20, help='epoch at which to start lr decay')
    parser.add_argument('--learning_rate_decay_every', type=int, default=50,
                        help='decay learning rate at every this number')

    parser.add_argument('--beam_size', type=int, default=3, help='default 3 in original paper')
    parser.add_argument('--beam_search', type=bool, default=True,
                        help='use beam search to inference the images')
    args = parser.parse_args()

    print('------------------------Model Inference Details--------------------------')
    print(args)

    # Start inference
    main(args)




