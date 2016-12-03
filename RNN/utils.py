import os
import collections
from six.moves import cPickle
import numpy as np
from operator import lshift, add

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
    return res.rstrip('\n').split(' ')


def load_data_and_labels(input_addr):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    input_file_handler = open(input_addr,'r')
    encoding = []
    y = []
    x_text = []
    for line in input_file_handler.readlines():
        labels = []
        linestripped = line.rstrip('\n').split(' ')
        if(len(linestripped) < 2):
            continue
        # Split by P and N, if want to split please comment the next line
        # x_text.append(linestripped[0])
        encoding_cleaned = linestripped[1:]
        encoding_cleaned.insert(0, " ")
        encoding_cleaned.append(" ")
        x_text.append(encoding_cleaned)
        for i in range(len(encoding_cleaned)):
            labels.append(linestripped[0])
        y.append(labels)
    input_file_handler.close()
    return [x_text, y]

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        word_file = os.path.join(data_dir, "word.pkl")
        label_file = os.path.join(data_dir, "label.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(word_file) and os.path.exists(label_file)  and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file,word_file, label_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(word_file, label_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, word_file, label_file, tensor_file):
        [words, labels] = load_data_and_labels(input_file)
        words_flat = [val for sublist in words for val in sublist]
        labels_flat = [val for sublist in labels for val in sublist]
        words_noduplicate = list(set(words_flat))
        labels_noduplicate = list(set(labels_flat))
        vocab = []
        for i in range(len(labels_noduplicate)):
            vocab.append(" "+ labels_noduplicate[i])
            for j in range(len(words_noduplicate)):
                vocab.append( words_noduplicate[j]+"+"+labels_noduplicate[i] )
        words_list = []
        for i in range(len(words_flat)):
            word_appended = words_flat[i]+"+"+labels_flat[i]
            words_list.append(word_appended)
        counter_vocab = collections.Counter(vocab)
        counter_label = collections.Counter(labels_flat)
        count_vocab_pairs = sorted(counter_vocab.items(), key=lambda x: -x[1])
        count_label_pairs = sorted(counter_label.items(), key=lambda x: -x[1])
        self.words, _ = zip(*count_vocab_pairs)
        self.labels, _ = zip(*count_label_pairs)
        self.vocab_size =  len(self.words)
        self.label_size =  len(self.labels)
        self.word_vocab = dict(zip(self.words, range(len(self.words))))
        self.label_vocab = dict(zip(self.labels, range(len(self.labels))))
        with open(word_file, 'wb') as f:
            cPickle.dump(self.words, f)
        with open(label_file, 'wb') as f:
            cPickle.dump(self.labels, f)
        self.tensor = np.array( list( map(self.word_vocab.get, words_list)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, word_file,label_file, tensor_file):
        with open(word_file, 'rb') as f:
            self.words = cPickle.load(f)
        with open(label_file, 'rb') as f:
            self.labels = cPickle.load(f)
        self.vocab_size = len(self.words)
        self.label_size = len(self.labels)
        self.word_vocab = dict(zip(self.words, range(len(self.words))))
        self.label_vocab = dict(zip(self.labels, range(len(self.labels))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tesor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
