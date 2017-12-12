# -*- coding:utf-8 -*-
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import string
import numpy as np
import nltk
import jieba
from PIL import Image
import datetime
import json
class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset = {}
        self.anns = []

        if not annotation_file == None:
            print('loading annotations into memory...')
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, 'r'))
            print(datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            # [{"caption_id":726786,"image_id":"b96ff46ba5b1cbe5bb4cc32b566431132ca71a64.jpg","caption":""光亮的房屋里一位背着包的女士旁边坐着一位穿着古装的男士""}]
            self.createIndex_AI_challenger()

    def createIndex_AI_challenger(self):
        print('creating index...')
        anns = {}
        for item in self.dataset:
            anns[str(item["caption_id"])] = {"caption": item["caption"], "image_id": item["image_id"]}
        print('index created!')
        self.anns = anns
        # print(self.anns)
        # {'52': {'caption': '餐厅里有一个穿着古装的男人在和一个右手拿着手机的女人拍照', 'image_id': '174b570c3ede32b07af7017b3b0389f38c130701.jpg'}, '28': {'caption': '两个手里戴着拳套的人在赛场上打拳击', 'image_id': '3111267a91c0d3c77418637d8f3bbe4bb9504cdd.jpg'},


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, image_dir, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """

        self.root = image_dir
        self.coco = COCO(json)

        self.ids = list(self.coco.anns.keys())
        # self.coco.anns
        # {'52': {'caption': '餐厅里有一个穿着古装的男人在和一个右手拿着手机的女人拍照', 'image_id': '174b570c3ede32b07af7017b3b0389f38c130701.jpg'}, '28': {'caption': '两个手里戴着拳套的人在赛场上打拳击', 'image_id': '3111267a91c0d3c77418637d8f3bbe4bb9504cdd.jpg'},

        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair ( image, caption, image_id )."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        #print(type(ann_id))
        #print(ann_id)
        caption = coco.anns[str(ann_id)]['caption']
        img_id = coco.anns[str(ann_id)]['image_id']

        #filename = coco.loadImgs(img_id)[0]['file_name']

        # if 'val' in filename.lower():
        #     path = 'val2014/' + filename
        # else:
        #     path = 'train2014/' + filename

        image = Image.open(os.path.join( self.root, img_id )).convert('RGB')
        if self.transform is not None:
            image = self.transform( image )

        # Convert caption (string) to word ids.
        tokens = jieba.cut(caption, cut_all=False)
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, img_id

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption) .
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
        img_ids: image ids in COCO dataset, for evaluation purpose
        ## filenames: image filenames in COCO dataset, for evaluation purpose
    """

    # Sort a data list by caption length (descending order).
    data.sort( key=lambda x: len( x[1] ), reverse=True )
    images, captions, img_ids = zip( *data ) # unzip

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    img_ids = list( img_ids )


    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths, img_ids


def get_loader(image_dir, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(image_dir,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
    # # Load vocabulary wrapper.
    # with open( args.vocab_path, 'rb') as f:
    #     vocab = pickle.load( f )
    #
    # # Build training data loader
    # data_loader = get_loader( args.image_dir, args.caption_path, vocab,
    #                           transform, args.batch_size,
    #                           shuffle=True, num_workers=args.num_workers )
