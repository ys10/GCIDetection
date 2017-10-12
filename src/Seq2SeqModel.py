#coding=utf-8
from __future__ import print_function

import logging
import os
import tensorflow as tf
import h5py
from tensorflow.contrib import rnn
from InputReader import InputReader
from ResultWriter import ResultWriter

''' Config the logger, output into log file.'''
log_file_name = "log/model.log"
if not os.path.exists(log_file_name):
    f = open(log_file_name, 'w')
    f.close()
logging.basicConfig(level = logging.DEBUG,
                format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=log_file_name,
                filemode='w')

''' Output to the console.'''
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

''' GPU  setting'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

''' HDF5 file (data) info.'''
hdf5DirPath = "data/hdf5/"
hdf5Extension = ".hdf5"
# Data file.
dataFilename = "APLAWDW"
dataFile = h5py.File(hdf5DirPath + dataFilename + hdf5Extension)
# Result file.
resultFilename = "APLAWDW_result"
resultFile = h5py.File(hdf5DirPath + resultFilename + hdf5Extension, "w")

'''Learning model parameters'''
learningRate = 1e-3
batchSize = 8 # During an iteration, each batch need memory space around 1Gb.
#  Total training iteration
iteration = 50000
#  After a fixed count of iteration, save output result of training.
saveIteration = 1000;
#  After a fixed count of iteration, display some info(eg. accuracy) about training.
displayIteration = 5

'''Data info'''
inputSize = 1
timestepSize = 84000
hiddenSize = 256
layerNum = 1
classNum = 2

'''DNN model'''
X = tf.placeholder(tf.float32, [batchSize, timestepSize, inputSize])
y = tf.placeholder(tf.float32, [batchSize, timestepSize, classNum])
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope("LSTM") as vs:
    cells = list()
    for _ in range (layerNum):
        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0)
        # Drop out in case of over-fitting.
        lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        # Stack same lstm cell
        cells.append(lstm_cell)
        pass
    stack = rnn.MultiRNNCell(cells)

outputs, _ = tf.nn.dynamic_rnn(stack, X, dtype=tf.float32)
logits = tf.contrib.layers.fully_connected(outputs, classNum, activation_fn=None)

# Loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
cost = tf.reduce_mean(cross_entropy)
train_op = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

# Evaluate
correct_pred = tf.equal(tf.argmax(logits, 2), tf.argmax(y, 2))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

'''Model save'''
# Initialize the saver to save session.
saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, max_to_keep=50)
saved_model_path = 'model/'
to_save_model_path = 'model/'

'''Start a session and run up.'''
with tf.Session(config=config) as sess:
    logging.info("Session started!")
    sess.run(tf.global_variables_initializer())
    # Prepare data set.
    dataSet = InputReader(dataFile, batchSize, timestepSize)
    # Prepare result writer.
    resultWriter = ResultWriter(resultFile)
    for i in range(iteration):
        (batchX, batchY) = dataSet.getBatch(i)
        _, trainingCost, modelOutput = sess.run([train_op, cost, logits], feed_dict={X:batchX, y: batchY, keep_prob: 1.0})
        logging.info("Iteration:" + str(i)
                     + ", \tbatch loss= {:.6f}".format(trainingCost))
        logging.debug("batchX:"+ str(batchX[0]))
        logging.debug("batchY:"+ str(batchY[0]))
        logging.debug("modelOutput:"+ str(modelOutput[0]))
        # Save output result.
        if (i)% saveIteration == 0:
            # resultWriter.saveBatchResult(modelOutput, dataSet.getBatchKeyList(i))
            saver.save(sess, to_save_model_path, global_step=epoch);
            logging.info("Model saved successfully to: " + to_save_model_path)
            # TODO
            pass
        # Display accuracy.
        if (i+1)% displayIteration == 0:
            train_logits, train_y, train_correct_pred, train_accuracy = sess.run([logits, y, correct_pred, accuracy], feed_dict={X:batchX, y: batchY, keep_prob: 1.0})
            logging.info("Epoch:" + str(dataSet.completedEpoch)
                         + ", \titeration:" + str(i)
                         + ", \tbatch loss= {:.6f}".format(trainingCost)
                         + ", \t training accuracy= {:.6f}".format(train_accuracy)
                         )
            logging.debug("Logits:" + str(train_logits)
                          + ", \ty:"+str(train_y)
                          + ", \tcorrect_pred:" + str(train_correct_pred)
                          )
            pass
        pass
    logging.info("Optimization Finished!")
    pass
pass
dataFile.close()
resultFile.close()