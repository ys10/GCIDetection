#coding=utf-8
from __future__ import print_function

import logging
import os
import tensorflow as tf
import h5py
from tensorflow.contrib import rnn
from DataSet import DataSet

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
iteration = 500
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
classNum = 1

'''DNN model'''
X = tf.placeholder(tf.float32, [batchSize, timestepSize, inputSize])
y = tf.placeholder(tf.float32, [batchSize, timestepSize, classNum])
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope("LSTM") as vs:
    # Define parameters of full connection between the second LSTM layer and output layer.
    # Define weights.
    weights = {
        'out': tf.Variable(tf.random_normal([hiddenSize, classNum]))
    }
    # Define biases.
    biases = {
        'out': tf.Variable(tf.random_normal([classNum]))
    }

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0)
    # Drop out in case of over-fitting.
    lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    # Stack two same lstm cell
    stack = rnn.MultiRNNCell([lstm_cell] * layerNum)

    # Define LSTM as a RNN.
    def RNN(x, weights, biases):
        outputs, _ = tf.nn.dynamic_rnn(stack, x, dtype=tf.float32)
        logits = tf.contrib.layers.fully_connected(outputs, classNum, activation_fn=None)
        return logits

    # Define prediction of RNN(LSTM).
    pred = RNN(X, weights, biases)

    # Loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
    cost = tf.reduce_mean(cross_entropy)
    train_op = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

    # Evaluate
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    '''Start a session and run up.'''
    with tf.Session(config=config) as sess:
        logging.info("Session started!")
        sess.run(tf.global_variables_initializer())
        # Prepare data set.
        dataSet = DataSet(dataFile, batchSize, timestepSize)
        for i in range(iteration):
            logging.info("Iteration: "+ str(i))
            (batchX, batchY) = dataSet.getBatch(i)
            if (i+1)% displayIteration == 0:
                train_accuracy = sess.run(accuracy, feed_dict={X:batchX, y: batchY, keep_prob: 1.0})
                logging.info("Epoch:%d,\t iteration:%d,\t training accuracy:%g" % ( dataSet.completedEpoch, (i+1), train_accuracy))
                pass
            sess.run(train_op, feed_dict={X:batchX, y: batchY, keep_prob: 1.0})
            pass
        pass