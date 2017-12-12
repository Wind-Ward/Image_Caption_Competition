# -*- coding:utf-8 -*-
import json
import torch
import torch.nn as nn
import numpy as np
import os
import glob
import pickle
from build_vocab_ai_challenger import Vocabulary
from torch.autograd import Variable 
from torchvision import transforms, datasets
#from coco.pycocotools.coco import COCO
#from coco.pycocoevalcap.eval import COCOEvalCap
from coco_caption.pycxtools.coco import COCO
from coco_caption.pycxevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt

# Variable wrapper
def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable( x, volatile=volatile )

# Show multiple images and caption words
#暂时没用
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image_original_11. Must have
            the same length as titles.
            
    Adapted from https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    """
    
    assert(( titles is None ) or (len( images ) == len( titles )))
    
    n_images = len( images )
    if titles is None: 
        titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
        
    fig = plt.figure( figsize=( 15, 15 ) )
    for n, (image, title) in enumerate( zip(images, titles) ):
        
        a = fig.add_subplot( np.ceil( n_images/ float( cols ) ), cols, n+1 )
        if image.ndim == 2:
            plt.gray()
            
        plt.imshow( image )
        a.axis('off')
        a.set_title( title, fontsize=200 )
        
    fig.set_size_inches( np.array( fig.get_size_inches() ) * n_images )
    
    plt.tight_layout( pad=0.4, w_pad=0.5, h_pad=1.0 )
    plt.show()




# MS COCO evaluation data loader
class CocoEvalLoader( datasets.ImageFolder ):

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
        # print(img_path)
        img = self.loader(os.path.join(self.root, img_path))
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, img_path.split(".")[0]

    def __len__(self):
        return len(self.file_list)

# MSCOCO Evaluation function
def coco_eval( model, args, epoch ):
    
    '''
    model: trained model to be evaluated
    args: pre-set parameters
    epoch: epoch #, for disp purpose
    '''
    
    model.eval()
    
    # Validation images are required to be resized to 224x224 already
    transform = transforms.Compose([ 
        transforms.Scale( (args.crop_size, args.crop_size) ),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load the vocabulary
    with open( args.vocab_path, 'rb' ) as f:
         vocab = pickle.load(f)
    
    # Wrapper the COCO VAL dataset
    eval_data_loader = torch.utils.data.DataLoader( 
        CocoEvalLoader(args.image_dir,transform),
        batch_size = args.eval_size, 
        shuffle = False, num_workers = args.num_workers,
        drop_last = False)
    
    # Generated captions to be compared with GT
    results = []
    print ('---------------------Start evaluation on MS-COCO dataset-----------------------')
    for i, (images,file_prefix) in enumerate(eval_data_loader):
        
        images = to_var(images)
        generated_captions, _, _ = model.sampler(images)
        #print(generated_captions)
        if torch.cuda.is_available():
            captions = generated_captions.cpu().data.numpy()
        else:
            captions = generated_captions.data.numpy()
        
        # Build caption based on Vocabulary and the '<end>' token
        for image_idx in range( captions.shape[0] ):
            
            sampled_ids = captions[image_idx]
            sampled_caption = []
            
            for word_id in sampled_ids:
                
                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                else:
                    sampled_caption.append(word)
            
            sentence = ''.join(sampled_caption)
            #生成的句子
            temp = {'image_id': file_prefix[image_idx], 'caption': sentence}
            results.append( temp )
        
        # Disp evaluation process
        if (i+1) % 10 == 0:
            print('[%d/%d]'%((i+1),len(eval_data_loader)))
            
            
    print('------------------------Caption Generated-------------------------------------')
            
    # Evaluate the results based on the COCO API
    result_path=os.path.join(args.model_path, "result")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    #print(result_path)
    resFile =result_path+"/"+ str(epoch) + '.json'
    json.dump(results,open( resFile , 'w' ),ensure_ascii=False,sort_keys=True, indent=2, separators=(',', ': '))

    def compute_m1(json_predictions_file, reference_file):
        """Compute m1_score"""
        m1_score = {}
        m1_score['error'] = 0
        cider = 0.
        try:
            coco = COCO(reference_file)
            coco_res = coco.loadRes(json_predictions_file)

            # create coco_eval object.
            coco_eval = COCOEvalCap(coco, coco_res)

            # evaluate results
            coco_eval.evaluate()
        except Exception:
            m1_score['error'] = 1
        else:
            # print output evaluation scores
            for metric, score in coco_eval.eval.items():
                print('%s: %.3f' % (metric, score))
                if metric == 'CIDEr':
                    cider = score
                #m1_score[metric] = score
        return cider

    print('-----------Evaluation performance on AI Challenger validation dataset for Epoch %d----------' % (epoch))
    cider=compute_m1(resFile,args.caption_val_json_path)
    print("Evaluation cider is %f" % cider)

    return cider


