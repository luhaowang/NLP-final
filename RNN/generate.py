from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

def clean_str(string):
    """
    split each encoding into many words.
    'word' represents a transistor connection, starting with 'P' or 'P'
    """
    res =""
    for i in range(len(string)):
        if(i!=0 and (string[i]=="P" or string[i]=="N")):
            res = res + " " + string[i]
        else:
            res += string[i]
    return res

def main():
    input_file_handler = open("./data/auto_train.txt",'r')
    output_file_handler = open("./output.txt",'w')
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save/',
                       help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=200,
                       help='number of words to sample')
    parser.add_argument('-rate', type=float, default=5.0,
                       help='rating')
    parser.add_argument('--prime', type=str, default='comment:',
                       help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    args = parser.parse_args()
    generate(args,input_file_handler,output_file_handler)
    input_file_handler.close()
    output_file_handler.close()
        

def generate(args,input_file_handler,output_file_handler):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'word.pkl'), 'rb') as f:
        words, word_vocab = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'label.pkl'), 'rb') as f:
        labels, label_vocab = cPickle.load(f)
    model = Model(saved_args, infer=True, pred=False)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(model.generate(sess,args.rate, words, word_vocab,labels, label_vocab, args.n, args.prime, args.sample))



if __name__ == '__main__':
    main()
