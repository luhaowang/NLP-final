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
    parser.add_argument('-n', type=int, default=7,
                       help='number of characters to sample')
    parser.add_argument('--prime', type=str, default=' Alice ',
                       help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    args = parser.parse_args()
    sample(args,input_file_handler,output_file_handler)
    input_file_handler.close()
    output_file_handler.close()
        

def sample(args,input_file_handler,output_file_handler):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'word.pkl'), 'rb') as f:
        words, word_vocab = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'label.pkl'), 'rb') as f:
        labels, label_vocab = cPickle.load(f)
    model = Model(saved_args, infer=False, pred=True)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #buffer_str = str(model.sample(sess, chars, chars_vocab, labels, labels_vocab, args.n, " Nn2n0n1Nn2n0n3Nn2n10n0Nn5n0n4Nn7n0n6Nn9n0n8Pn0n10n11Pn10n0n11 ", args.sample))
            #output_file_handler.write(buffer_str+'\n')
            hsh_total = {}
            hsh_correct = {}
            for i in range(len(labels)):
                hsh_correct[labels[i]] = 0
                hsh_total[labels[i]] = 0
            sumNum = 0
            RightNum = 0
            for line in input_file_handler.readlines():
                if(len(line.rstrip('\n').split(' '))>2):
                    linesplitted = line.rstrip('\n').split(' ')
                    sumNum += 1
                    hsh_total[linesplitted[0]] += 1
                    encoding_cleaned = linesplitted[1:]
                    encoding_cleaned.insert(0," ")
                    encoding_cleaned.append(" ")
                    output_file_handler.write(" ".join(linesplitted[1:]))
                    buffer_str = str(model.sample(sess, words, word_vocab,labels, label_vocab, args.n, encoding_cleaned, args.sample))
                    if(linesplitted[0] == buffer_str):
                        RightNum += 1
                        hsh_correct[linesplitted[0]] += 1
                    print(linesplitted[0]+":"+buffer_str+"\t accuracy: "+ str(float(float(RightNum)/float(sumNum)))+'\n')
                    output_file_handler.write("\t" + buffer_str+"\t accuracy: "+ str(float(float(RightNum)/float(sumNum)))+'\n')
                else:
                    continue
            print("Total number of test examples: {}".format(sumNum))
            print("Accuracy: {:g}".format(float(RightNum)/float(sumNum)))
            print("Accuracy details: \n ")
            for k in range(len(labels)):
                if float(hsh_total[labels[k]]) != 0:
                    print("Accuracy for {} : {:g}".format(labels[k], float(hsh_correct[labels[k]])/float(hsh_total[labels[k]])))
            

if __name__ == '__main__':
    main()
