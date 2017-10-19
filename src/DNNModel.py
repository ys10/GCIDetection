import os
import h5py
import logging
import tensorflow as tf
from tensorflow.contrib import rnn
from src.InputReader import InputReader
from src.ResultWriter import ResultWriter


''' Config the logger, output into log file.'''
log_file_name = "log/model.log"
if not os.path.exists(log_file_name):
    f = open(log_file_name, 'w')
    f.close()
logging.basicConfig(level = logging.INFO,
                format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
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

    def __init__(self, inputSize, timeStepSize, hiddenSize, layerNum, classNum, learningRate = 1e-3):
        self.inputSize = inputSize
        self.timeStepSize = timeStepSize
        self.hiddenSize = hiddenSize
        self.layerNum = layerNum
        self.classNum = classNum
        self.learningRate = learningRate
        '''Place holder'''
        self.x = tf.placeholder(tf.float32, [None, None, self.inputSize])
        self.y = tf.placeholder(tf.float32, [None, None, self.classNum])
        self.keep_prob = tf.placeholder(tf.float32)
        '''DNN model'''
        self.__initDNNModel()
        self.outputs, _ = tf.nn.dynamic_rnn(self.stack, self.x, dtype=tf.float32)
        self.logits = tf.contrib.layers.fully_connected(self.outputs, self.classNum, activation_fn=None)
        '''Loss function'''
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        # self.cost = tf.reduce_mean(ctc_ops.ctc_loss(labels=self.y, inputs=self.logits, sequence_length=self.timestepSize, time_major=False))
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.cost)
        '''Evaluate'''
        self.correct_pred = tf.equal(tf.argmax(self.logits, 2), tf.argmax(self.y, 2))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        '''Model save'''
        # Initialize the saver to save session.
        self.saver = tf.train.Saver(max_to_keep=50)
        self.saved_model_path = 'model/'
        self.to_save_model_path = 'model/'
        ''' GPU  setting'''
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8
        pass


    def __initDNNModel(self):
        cells = list()
        for _ in range(self.layerNum):
            # Define a lstm cell with tensorflow
            lstm_cell = rnn.BasicLSTMCell(self.hiddenSize, forget_bias=1.0)
            # Drop out in case of over-fitting.
            lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            # Stack same lstm cell
            cells.append(lstm_cell)
            pass
        self.stack = rnn.MultiRNNCell(cells)
        pass


    def setDataFilename(self, dataFilename, resultFilename):
        self.dataFilename = dataFilename
        self.resultFilename = resultFilename
        pass


    def __openDataFile(self):
        self.dataFile = h5py.File(self.dataFilename)
        self.resultFile = h5py.File(self.resultFilename)
        pass


    def __closeDataFile(self):
        self.dataFile.close()
        self.resultFile.close()
        pass


    def train(self, trainIteration = 10000, saveIteration = 100, displayIteration = 5, batchSize = 4):
        #  Total training iteration
        self.trainIteration = trainIteration
        #  After a fixed count of iteration, save output result of training.
        self.saveIteration = saveIteration
        #  After a fixed count of iteration, display some info(eg. accuracy) about training.
        self.displayIteration = displayIteration
        #  Batch size
        self.batchSize = batchSize
        # Start a session and run up.
        with tf.Session(config=self.config) as sess:
            logging.info("Session started!")
            sess.run(tf.global_variables_initializer())
            self.__openDataFile()
            # Prepare data set.
            dataSet = InputReader(self.dataFile, self.batchSize, self.timeStepSize)
            # Prepare result writer.
            resultWriter = ResultWriter(self.resultFile)
            for i in range(self.trainIteration):
                (batchX, batchY) = dataSet.getBatch(i)
                _, trainingCost, modelOutput = sess.run([self.optimizer, self.cost, self.logits],
                                                        feed_dict={self.x: batchX, self.y: batchY, self.keep_prob: 1.0})
                logging.info("Iteration:" + str(i)
                             + ", \tbatch loss= {:.6f}".format(trainingCost))
                logging.debug("batchX:" + str(batchX[0]))
                logging.debug("batchY:" + str(batchY[0]))
                logging.debug("modelOutput:" + str(modelOutput[0]))
                # Save output result.
                if (i) % self.saveIteration == 0:
                    # Save model
                    self.saver.save(sess, self.to_save_model_path, global_step=self.saveIteration);
                    logging.info("Model saved successfully to: " + self.to_save_model_path)
                    # Save output
                    keyList = dataSet.getBatchKeyList(i);
                    resultWriter.saveBatchResult(modelOutput, keyList)
                    logging.info("Model output saved successfully:")
                    logging.info("Keys of saved model outputs:" + str(keyList))
                    pass
                # Display accuracy.
                if (i + 1) % self.displayIteration == 0:
                    train_logits, train_y, train_correct_pred, train_accuracy = sess.run(
                        [self.logits, self.y, self.correct_pred, self.accuracy], feed_dict={self.x: batchX, self.y: batchY, self.keep_prob: 1.0})
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

    pass
