import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from operator import rshift, lshift
import numpy as np
from copy import deepcopy
from math import log

class Model():
    def __init__(self, args, infer=False, pred=False):
        self.args = args
        if pred or infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, words, word_vocab, labels, label_vocab, num=200, prime=' VDD ', sampling_type=1):
        vocab_size = len(word_vocab)
        #parsed from process_prob.py
        dic = {'5.0':0.2, '4.0':0.2, '3.0':0.2, '2.0':0.2, '1.0':0.2}
        p = np.zeros((len(labels),len(prime)))
        for l in range(len(labels)):
            state = sess.run(self.cell.zero_state(1, tf.float32))
            probs = np.zeros( (len(prime),vocab_size))
            lbs = labels[l]
            for chi in range(len(prime)):
                word = prime[chi]
                x = np.zeros((1, 1))
                key = word+"+"+lbs
                x[0,0] = word_vocab[key]
                #print x
                feed = {self.input_data: x, self.initial_state:state}
                [probs[chi], state] = sess.run([self.probs, self.final_state], feed)
                if chi < len(prime):
                    p[l][chi] = log(probs[chi][word_vocab[word+"+"+lbs]])+log(dic[lbs])
                    
        #print p
        #print labels[np.argmax(p[:,-1])]
        return labels[np.argmax(p[:,-1] + p[:,-2])]
    
    def generate(self, sess,rate,words, word_vocab, labels, label_vocab, num=200, prime=' VDD ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        if not len(prime) or prime == " ":
            prime  = random.choice(list(word_vocab.keys()))    
        print ("rating: {}".format(rate))
        for word in prime.split()[:-1]:
            #print (word)
            x = np.zeros((1, 1))
            x[0, 0] = word_vocab.get(word,0)
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)
         
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        word = prime.split()[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = word_vocab.get(word+"+"+str(rate),0)
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if word == '\n':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = words[sample].split("+")[0]
            ret += ' ' + pred
            word = pred
        return ret


