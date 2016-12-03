import numpy as np
import re
import itertools
from collections import Counter


def clean_str(res):
  
    """
    split each encoding into many words.
    'word' represents a transistor connection, starting with 'P' or 'P'
    """
    return res.strip(" ")


def load_data_and_labels(input_addr):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    dic_coding = {}
    dic_coding_inv = {}
    input_file_handler = open(input_addr,'r')
    encoding = []
    labels = []
    sequence = 0
    for line in input_file_handler.readlines():
        linestripped = line.rstrip('\n').split(' ')
        if(dic_coding.has_key(linestripped[0]) == False):
            dic_coding[linestripped[0]] = sequence
            dic_coding_inv[sequence] = linestripped[0]
            sequence += 1
    input_file_handler.close()
    num = len(dic_coding)
    input_file_handler = open(input_addr,'r')
    for line in input_file_handler.readlines():
        linestripped = line.rstrip('\n').split(' ')
        each_coding = [0]* num
        each_coding[dic_coding[linestripped[0]]] = 1
        encoding.append(" ".join(linestripped[1:]))
        labels.append(each_coding)
    input_file_handler.close()
    x_text = encoding
    # Split each word 
    x = [clean_str(sent) for sent in x_text]
    # Generate labels
    y = np.array(labels)
    print "data is loaded!! {:} \n".format(input_addr)
    return [x, y,dic_coding, dic_coding_inv]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
