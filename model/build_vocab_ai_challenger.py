# coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import jieba
import json
import pickle
import argparse
import datetime
from collections import Counter

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
            self.createIndex_AI_challenger()

    #[{"caption_id":726786,"image_id":"b96ff46ba5b1cbe5bb4cc32b566431132ca71a64.jpg","caption":""光亮的房屋里一位背着包的女士旁边坐着一位穿着古装的男士""}]
    def createIndex_AI_challenger(self):
        print('creating index...')
        anns={}
        for item in self.dataset:
            anns[str(item["caption_id"])]={"caption":item["caption"],"image_id":item["image_id"]}
        print('index created!')
        self.anns=anns
        #print(self.anns)
        #{'52': {'caption': '餐厅里有一个穿着古装的男人在和一个右手拿着手机的女人拍照', 'image_id': '174b570c3ede32b07af7017b3b0389f38c130701.jpg'}, '28': {'caption': '两个手里戴着拳套的人在赛场上打拳击', 'image_id': '3111267a91c0d3c77418637d8f3bbe4bb9504cdd.jpg'},


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        #tokens = nltk.tokenize.word_tokenize(caption.lower())
        tokens=jieba.cut(str(coco.anns[id]["caption"]), cut_all=False)
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(json=args.caption_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)
    # print("word2idx")
    # print(vocab.word2idx)
    # print("idx2word")
    # print(vocab.idx2word)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='/Users/yinfeng/PycharmProjects/AdaptiveAttentionCaption/data/adaptive_attention_format_train_11.json',
                        #default='/home/windward/symbolic_yangyi/PycharmProjects/AdaptiveAttentionCAption/data/adaptive_attention_format_train_237000.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str,
                        default='./vocab_ai_challenger_11.pkl',
                        #default='/home/windward/symbolic_yangyi/PycharmProjects/AdaptiveAttentionCAption/data/vocab_ai_challenger_237000.pkl',
                        help='path for saving vocabulary wrapper')


    parser.add_argument('--threshold', type=int, default=5,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
