import logging
import os
from abc import abstractmethod

import h5py
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from dataAccessor.InputReader import InputReader
from dataAccessor.ResultWriter import ResultWriter

''' Config the logger, output into log file.'''
log_file_name = "log/model.log"
if not os.path.exists(log_file_name):
    f = open(log_file_name, 'w')
    f.close()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_file_name,
                    filemode='w')

''' Output to the console.'''
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


class DNNModel(object):
    def __init__(self, inputSize, timeStepSize, hiddenSize, layerNum, classNum, learningRate=1e-3):
        with tf.name_scope('ModelParameter'):
            self.inputSize = inputSize
            self.timeStepSize = timeStepSize
            self.hiddenSize = hiddenSize
            self.layerNum = layerNum
            self.classNum = classNum
            self.learningRate = learningRate
            self.__variableSummaries(self.learningRate)
            pass
        '''Place holder'''
        with tf.name_scope('PlaceHolder'):
            self.x = tf.placeholder(tf.float32, [None, None, self.inputSize])
            self.y = tf.placeholder(tf.float32, [None, None, self.classNum])
            self.keep_prob = tf.placeholder(tf.float32)
            pass
        '''DNN model'''
        self.__DNNModel()
        '''Loss function'''
        self.__lossFunction()
        '''Optimizer'''
        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.cost)
            pass
        '''Evaluator'''
        with tf.name_scope('Evaluator'):
            self.correct_pred = tf.equal(tf.argmax(self.logits, 2), tf.argmax(self.y, 2))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            self.__variableSummaries(self.accuracy)
            pass
        '''Model save'''
        # Initialize the saver to save session.
        self.saver = tf.train.Saver(max_to_keep=50)
        self.modelRestorePath = None
        self.modelSavePath = 'model/'
        ''' GPU  setting'''
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8
        pass

    def __DNNModel(self):
        with tf.name_scope('DNNModel'):
            lstm_fw_cells = list()
            lstm_bw_cells = list()
            for _ in range(self.layerNum):
                # Define LSTM cells with tensorflow
                fw_cell = rnn.BasicLSTMCell(self.hiddenSize, forget_bias=1.0)
                bw_cell = rnn.BasicLSTMCell(self.hiddenSize, forget_bias=1.0)
                # Drop out in case of over-fitting.
                fw_cell = rnn.DropoutWrapper(fw_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
                bw_cell = rnn.DropoutWrapper(bw_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
                # Stack same LSTM cells.
                lstm_fw_cells.append(fw_cell)
                lstm_bw_cells.append(bw_cell)
                pass
            outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                lstm_fw_cells,
                lstm_bw_cells,
                self.x,
                dtype=tf.float32
            )
            self.logits = tf.contrib.layers.fully_connected(outputs, self.classNum, activation_fn=None)
            self.__variableSummaries(self.logits)
            pass
        pass

    def __lossFunction(self):
        with tf.name_scope('LossFunction'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
            self.cost = tf.reduce_mean(cross_entropy)
            self.__variableSummaries(self.cost)
            pass
        pass

    def setDataFilename(self, dataFilename, resultFilename):
        self.dataFilename = dataFilename
        self.resultFilename = resultFilename
        pass

    def setModelSavePath(self, modelRestorePath=None, modelSavePath=None):
        self.modelRestorePath = modelRestorePath
        self.modelSavePath = modelSavePath
        pass

    def setSummarySavePath(self, summarySavePath):
        self.summarySavePath = summarySavePath
        pass

    def __openDataFile(self):
        self.dataFile = h5py.File(self.dataFilename)
        self.resultFile = h5py.File(self.resultFilename)
        pass

    def __closeDataFile(self):
        self.dataFile.close()
        self.resultFile.close()
        pass

    def __variableSummaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
            pass
        pass

    def train(self, trainIteration=10000, saveIteration=100, displayIteration=5, batchSize=4, samplingRate=20000):
        #  Total training iteration
        self.trainIteration = trainIteration
        #  After a fixed count of iteration, save output result of training.
        self.saveIteration = saveIteration
        #  After a fixed count of iteration, display some info(eg. accuracy) about training.
        self.displayIteration = displayIteration
        #  Batch size
        self.batchSize = batchSize
        #  Sampling rate
        self.samplingRate = samplingRate
        # Start a session and run up.
        with tf.Session(config=self.config) as sess:
            logging.info("Training session started!")
            sess.run(tf.global_variables_initializer())
            '''Restore model.'''
            if self.modelRestorePath is not None:
                self.saver.restore(sess, self.modelRestorePath)
                logging.info("Model restored from:"+str(self.modelRestorePath))
                pass
            # Summary
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.summarySavePath, sess.graph)
            # Prepare data set.
            self.__openDataFile()
            dataSet = InputReader(self.dataFile, self.batchSize, self.timeStepSize)
            # Prepare result writer.
            resultWriter = ResultWriter(self.resultFile, self.samplingRate)
            for i in range(self.trainIteration):
                (batchX, batchY) = dataSet.getBatch(i)
                summary, _, trainingCost, modelOutput = sess.run([merged, self.optimizer, self.cost, self.logits],
                                                        feed_dict={self.x: batchX, self.y: batchY, self.keep_prob: 1.0})
                logging.info("Iteration:" + str(i)
                             + ", \tbatch loss= {:.6f}".format(trainingCost))
                logging.debug("batchX:" + str(batchX[0]))
                logging.debug("batchY:" + str(batchY[0]))
                logging.debug("modelOutput:" + str(modelOutput[0]))
                # Add summary.
                train_writer.add_summary(summary, global_step=i)
                # Save output result.
                if (i) % self.saveIteration == 0:
                    # Save model
                    self.saver.save(sess, self.modelSavePath, global_step=self.saveIteration)
                    logging.info("Model saved successfully to: " + self.modelSavePath)
                    # Save output
                    keyList = dataSet.getBatchKeyList(i)
                    resultWriter.saveBatchResult(modelOutput, keyList)
                    logging.info("Model output saved successfully:")
                    logging.info("Keys of saved model outputs:" + str(keyList))
                    pass
                # Display accuracy.
                if (i + 1) % self.displayIteration == 0:
                    train_logits, train_y, train_correct_pred, train_accuracy = sess.run(
                        [self.logits, self.y, self.correct_pred, self.accuracy],
                        feed_dict={self.x: batchX, self.y: batchY, self.keep_prob: 1.0})
                    logging.info("Epoch:" + str(dataSet.completedEpoch)
                                 + ", \titeration:" + str(i)
                                 + ", \tbatch loss= {:.6f}".format(trainingCost)
                                 + ", \t training accuracy= {:.6f}".format(train_accuracy)
                                 )
                    logging.debug("Logits:" + str(train_logits)
                                  + ", \ty:" + str(train_y)
                                  + ", \tcorrect_pred:" + str(train_correct_pred)
                                  )
                    pass
                pass
            self.__closeDataFile()
            logging.info("Optimization Finished!")
            pass
        pass

    def test(self, samplingRate=20000):
        #  Sampling rate
        self.samplingRate = samplingRate
        # Start a session and run up.
        with tf.Session(config=self.config) as sess:
            logging.info("Testing session started!")
            '''Restore model.'''
            if self.modelRestorePath is not None:
                self.saver.restore(sess, self.modelRestorePath)
                logging.info("Model restored from:"+str(self.modelRestorePath))
            else:
                logging.info("Model restore failed.")
                return
            '''Prepare testing parameters.'''
            self.__openDataFile()
            # Prepare data set.
            dataSet = InputReader(dataFile=self.dataFile, batchSize=1, maxTimeStep=self.timeStepSize)
            # Prepare result writer.
            resultWriter = ResultWriter(self.resultFile, self.samplingRate)
            # Test iteration.
            testIteration = dataSet.getBatchCount()
            '''Forward testing data.'''
            for i in range(testIteration):
                (batchX, batchY) = dataSet.getBatch(i)
                modelOutput = sess.run([self.logits], feed_dict={self.x: batchX, self.y: batchY, self.keep_prob: 1.0})
                # Save output
                keyList = dataSet.getBatchKeyList(i)
                resultWriter.saveBatchResult(modelOutput, keyList)
                pass
            pass
        logging.info("Testing finished!")
        pass

    pass
