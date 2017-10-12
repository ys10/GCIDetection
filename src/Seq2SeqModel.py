#coding=utf-8
from __future__ import print_function

import logging
import os
import tensorflow as tf
import h5py
from tensorflow.contrib import rnn
from tensorflow.python.ops import ctc_ops
from InputReader import InputReader


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
hdf5Filename = "APLAWDW_s_01_a"
hdf5Extension = ".hdf5"
dataFile = h5py.File(hdf5DirPath + hdf5Filename + hdf5Extension)

'''Learning model parameters'''
learningRate = 1e-3
batchSize = 8 # During an iteration, each batch need memory space around 1Gb.
#  Total training iteration
iteration = 50
#  After a fixed count of iteration, display some info(eg. accuracy) about training.
displayIteration = 5

'''Data info'''
# 每个时刻的输入特征是40000维的，就是每个时刻输入一行，一行有 1 个像素
inputSize = 1
# 时序持续长度为40000，即每做一次预测，需要先输入40000行
timestepSize = 20480
# 每个隐含层的节点数
hiddenSize = 256
# LSTM layer 的层数
layerNum = 1
# 最后输出分类类别数量，如果是回归预测的话应该是 1
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
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=logits, pos_weight=1000)
    cost = tf.reduce_mean(cross_entropy)
    # cost = tf.reduce_mean(ctc_ops.ctc_loss(labels=y, inputs=logits, sequence_length=timestepSize, time_major=False))
    train_op = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

    # Evaluate
    correct_pred = tf.equal(tf.argmax(logits, 2), tf.argmax(y, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    '''Start a session and run up.'''
    with tf.Session(config=config) as sess:
        logging.info("Session started!")
        sess.run(tf.global_variables_initializer())
        # Prepare data set.
        inputReader = InputReader(dataFile, batchSize, timestepSize)
        for i in range(iteration):
            (batchX, batchY) = inputReader.getBatch(i)
            _, trainingCost, modelOutput = sess.run([train_op, cost, logits], feed_dict={X:batchX, y: batchY, keep_prob: 1.0})
            logging.info("Iteration:" + str(i)
                         + ", \tbatch loss= {:.6f}".format(trainingCost))
            logging.debug("batchX:"+ str(batchX[0]))
            logging.debug("batchY:"+ str(batchY[0]))
            logging.debug("modelOutput:"+ str(modelOutput[0]))
            # Display accuracy.
            if (i+1)% displayIteration == 0:
                train_logits, train_y, train_correct_pred, train_accuracy = sess.run([logits, y, correct_pred, accuracy], feed_dict={X:batchX, y: batchY, keep_prob: 1.0})
                logging.info("Epoch:" + str(inputReader.completedEpoch)
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