#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1470976594/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

testfile = "./data/encoding3/v2/encoding3v2_2L_multiple_labels_nonetype_regression_afterpreprocess_haveduplicate_input_train.txt" 
trainfile = "./data/encoding3/v2/encoding3v2_2L_multiple_labels_nonetype_regression_afterpreprocess_haveduplicate_input_train.txt"  
# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test, dic_coding, dic_coding_inv = data_helpers.load_data_and_labels(testfile)
    x_pred, y_pred, dic_coding_pred, dic_coding_inv_pred = data_helpers.load_data_and_labels(trainfile)
    y_test = np.argmax(y_test, axis=1)
else:
    x_before = ["Nn2n0n1Nn2n0n4Nn2n1n0Nn2n1n0Pn3n0n1Pn3n0n4Pn3n1n0Pn3n1n0"]
    x_raw = [ data_helpers.clean_str(sent) for sent in x_before]
    y_test = [[1,0,0,0,0, 0,0,0,0,0, 0]]
    y_test = np.argmax(y_test, axis=1)

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    j = 0
    hsh_total = {}
    hsh_correct = {}
    for i in range(len(dic_coding)):
        hsh_correct[i] = 0
        hsh_total[i] = 0
    print all_predictions
    print y_test
    correct_count = 0
    for i in range(len(y_test)):
        hsh_total[y_test[i]] += 1
        if dic_coding_inv_pred[all_predictions[i]] == dic_coding_inv[y_test[i]]:
            hsh_correct[y_test[i]] += 1
            correct_count += 1
    correct_predictions = float(correct_count)
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    print "Accuracy details: "
    for k, v in dic_coding.items():
        print("Accuracy for {} : {:g}".format(k, float(hsh_correct[v])/float(hsh_total[v])))

