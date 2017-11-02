import logging
import os

import h5py
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from dataAccessor.InputReader import *

''' Config the logger, output into log file.'''
log_file_name = "log/model.log"
if not os.path.exists(log_file_name):
    f = open(log_file_name, 'w')
    f.close()
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_file_name,
                    filemode='w')

''' Output to the console.'''
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


class DNNModel(object):
    def __init__(self, inputSize, timeStepSize, hiddenSize, layerNum, outputSize, classNum, learningRate=1e-3):
        with tf.name_scope('ModelParameter'):
            self.inputSize = inputSize
            self.timeStepSize = timeStepSize
            self.hiddenSize = hiddenSize
            self.layerNum = layerNum
            self.outputSize = outputSize
            self.classNum = classNum
            # self.learningRate = learningRate
            self.globalStep = tf.Variable(0, trainable=False, dtype=tf.int32)
            self.decaySteps= 10
            self.decayRate = 0.5
            self.learningRate = tf.train.exponential_decay(learningRate, self.globalStep, self.decaySteps, self.decayRate)
            self.variableSummaries(self.learningRate)
            pass
        '''Place holder'''
        with tf.name_scope('PlaceHolder'):
            self.x = tf.placeholder(tf.float32, [None, None, self.inputSize])# X shape: (batchSize, timeSteps, inputSize)
            self.y = tf.placeholder(tf.float32, [None, None, self.outputSize])# Y shape: (batchSize, timeSteps, outputSize)
            # self.maskMatrix = tf.placeholder(tf.float32, [None, None, None]) # Mask shape: (batchSize, timeSteps(nGCIs), timeSteps)
            self.maskVector = tf.placeholder(tf.bool, [None, None]) # Mask shape: (batchSize, timeSteps)
            self.gciCount = tf.placeholder(tf.float32, [None, None])  # GCI count shape: (batchSize, 1)
            self.keep_prob = tf.placeholder(tf.float32)
            pass
        '''DNN model'''
        self.DNNModel()
        '''Loss function'''
        self.loss_weights = tf.reshape(tf.Variable([0.3, 0.7]), shape=(2, 1))
        self.lossFunction()
        '''Optimizer'''
        with tf.name_scope('Optimizer'):
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.cost)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
            params = tf.trainable_variables()
            gradients = tf.gradients(self.cost, params)
            clippedGradients, gradNorm = tf.clip_by_global_norm(gradients, 5)
            self.update = self.optimizer.apply_gradients(zip(clippedGradients, params), global_step=self.globalStep)
            pass
        '''Evaluator'''
        self.evaluator()
        '''Model save'''
        # Initialize the saver to save session.
        self.saver = tf.train.Saver(max_to_keep=50)
        self.modelRestorePath = None
        self.modelSavePath = 'model/'
        ''' GPU  setting'''
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        # self.config.gpu_options.per_process_gpu_memory_fraction = 0.8
        pass

    def DNNModel(self):
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
            self.variableSummaries(self.logits)
            pass
        pass

    def lossFunction(self):
        with tf.name_scope('LossFunction'):
            #TODO
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
            self.cost = tf.reduce_mean(cross_entropy)
            self.variableSummaries(self.cost)
            pass
        pass

    def evaluator(self):
        with tf.name_scope('Evaluator'):
            self.correct_pred = tf.equal(tf.argmax(self.logits, 2), tf.argmax(self.y, 2))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            self.variableSummaries(self.accuracy)
            pass
        pass

    def setTrainingDataFilename(self, trainingDataFilename, validationDataFilename):
        self.trainingDataFilename = trainingDataFilename
        self.validationDataFilename = validationDataFilename
        pass

    def setTestingDataFilename(self, testingDataFilename, resultFilename):
        self.testingDataFilename = testingDataFilename
        self.resultFilename = resultFilename
        pass

    def setResultWriter(self, resultWriter):
        self.resultWriter = resultWriter
        pass

    def setModelSavePath(self, modelRestorePath=None, modelSavePath=None):
        self.modelRestorePath = modelRestorePath
        self.modelSavePath = modelSavePath
        pass

    def setSummarySavePath(self, summarySavePath):
        self.summarySavePath = summarySavePath
        pass

    def openTrainingDataFile(self):
        self.trainingDataFile = h5py.File(self.trainingDataFilename)
        self.validationDataFile = h5py.File(self.validationDataFilename)
        pass

    def openTestingDataFile(self):
        self.testingDataFile = h5py.File(self.testingDataFilename)
        self.resultFile = h5py.File(self.resultFilename)
        pass

    def closeTrainingDataFile(self):
        self.trainingDataFile.close()
        self.validationDataFile.close()
        pass

    def closeTestingDataFile(self):
        self.testingDataFile.close()
        self.resultFile.close()
        pass

    def setDataFileInfo(self, samplingRate=20000, frameSize=9, frameStride=9):
        self.samplingRate = samplingRate
        self.frameSize = frameSize
        self.frameStride = frameStride
        pass

    def variableSummaries(self, var):
        var = tf.cast(var, dtype=tf.float32)
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

    def train(self, trainIteration=10000, saveIteration=100, displayIteration=5, batchSize=4):
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
            self.openTrainingDataFile()
            trainingDataSet = InputReader(self.trainingDataFile, self.batchSize, self.timeStepSize)
            for i in range(self.trainIteration):
                (batchX, batchY) = trainingDataSet.getBatch(i)
                summary, _, trainingCost = sess.run([merged, self.update, self.cost],
                                                        feed_dict={self.x: batchX, self.y: batchY, self.keep_prob: 1.0})
                logging.info("Iteration:" + str(i)
                             + ", \tbatch loss= {:.6f}".format(trainingCost))
                logging.debug("batchX:" + str(batchX[0]))
                logging.debug("batchY:" + str(batchY[0]))
                # Add summary.
                train_writer.add_summary(summary, global_step=i)
                # Save output result.
                if (i) % self.saveIteration == 0:
                    # Save model
                    self.saver.save(sess, self.modelSavePath, global_step=self.saveIteration)
                    logging.info("Model saved successfully to: " + self.modelSavePath)
                    pass
                # Display accuracy.
                if (i + 1) % self.displayIteration == 0:
                    train_logits, train_y, train_correct_pred, train_accuracy = sess.run(
                        [self.logits, self.y, self.correct_pred, self.accuracy],
                        feed_dict={self.x: batchX, self.y: batchY, self.keep_prob: 1.0})
                    logging.info("Epoch:" + str(trainingDataSet.completedEpoch)
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
            self.closeTrainingDataFile()
            logging.info("Optimization Finished!")
            pass
        pass

    def test(self):
        #  Set result file of result writer.
        self.resultWriter.setResultFile(self.resultFile)
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
            self.openTestingDataFile()
            # Prepare data set.
            testDataSet = InputReader(dataFile=self.testingDataFile, batchSize=1, maxTimeStep=self.timeStepSize)
            # Test iteration.
            testIteration = testDataSet.getBatchCount()
            '''Forward testing data.'''
            for i in range(testIteration):
                (batchX, batchY, batchGCICount) = testDataSet.getBatch(i)
                modelOutput = sess.run([tf.nn.softmax(self.logits)], feed_dict={self.x: batchX, self.y: batchY, self.gciCount: batchGCICount, self.keep_prob: 1.0})
                # Save output
                keyList = testDataSet.getBatchKeyList(i)
                self.resultWriter.saveBatchResult(modelOutput, keyList)
                pass
            pass
            self.closeTestingDataFile()
        logging.info("Testing finished!")
        pass

    pass
