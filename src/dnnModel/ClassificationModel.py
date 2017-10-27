# import tensorflow.contrib.rnn as rnn
from dnnModel.DNNModel import *
from dataAccessor.ResultWriter import *


class ClassificationModel(DNNModel):
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
            rnnOutputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                lstm_fw_cells,
                lstm_bw_cells,
                self.x,
                dtype=tf.float32
            )
            self.linearOutput = tf.contrib.layers.fully_connected(rnnOutputs, self.outputSize, activation_fn=None)
            with tf.name_scope('logits'):
                self.logits = tf.nn.softmax(self.linearOutput)
                self.variableSummaries(self.logits)
                pass
            with tf.name_scope('results'):
                self.results = tf.argmin(self.logits, 2)
                self.estimatedGCICount = tf.reduce_sum(self.results)
                pass
            pass
        pass

    def lossFunction(self):
        with tf.name_scope('LossFunction'):
            with tf.name_scope('difference'):
                self.difference = tf.subtract(self.logits, self.y)
            with tf.name_scope('costDistribute'):
                self.costDistribute = tf.matmul(self.maskMatrix, self.difference)
                # self.costDistribute = tf.boolean_mask( self.difference, self.mask)
            with tf.name_scope('larynxCycleCost'):
                self.larynxCycleCost = tf.reduce_sum(tf.square(self.costDistribute))
                self.variableSummaries(self.larynxCycleCost)
                pass
            with tf.name_scope('outOfLarynxCycleCost'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.linearOutput)
                self.outOfLarynxCycleCost = tf.reduce_mean(cross_entropy)
                self.variableSummaries(self.outOfLarynxCycleCost)
                pass
            with tf.name_scope('cost'):
                # costWeights = tf.nn.softmax(tf.Variable(tf.random_normal([2, 1])))
                # self.cost = tf.reduce_sum(tf.matmul(tf.stack([[self.larynxCycleCost, self.outOfLarynxCycleCost]]), costWeights))
                self.cost = self.outOfLarynxCycleCost
                # self.cost = self.larynxCycleCost + 1e3 * self.outOfLarynxCycleCost
                self.variableSummaries(self.cost)
                pass
            pass
        pass


    # def lossFunction(self):
    #     with tf.name_scope('LossFunction'):
    #         with tf.name_scope('mask'):
    #             with tf.name_scope('maskedLabels'):
    #                 self.maskedLabels = tf.boolean_mask(self.y, self.maskVector)
    #                 self.variableSummaries(self.maskedLabels)
    #                 pass
    #             with tf.name_scope('maskedLogits'):
    #                 self.maskedLogits = tf.boolean_mask(self.logits, self.maskVector)
    #                 self.variableSummaries(self.maskedLogits)
    #                 pass
    #             pass
    #         with tf.name_scope('larynxCost'):
    #             cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.maskedLabels,
    #                                                                     logits=self.maskedLogits)
    #             self.larynxCycleCost = tf.reduce_mean(cross_entropy)
    #             self.variableSummaries(self.larynxCycleCost)
    #             pass
    #         with tf.name_scope('outOfLarynxCycleCost'):
    #             cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
    #             self.outOfLarynxCycleCost = tf.reduce_mean(cross_entropy)
    #             self.variableSummaries(self.outOfLarynxCycleCost)
    #             pass
    #         with tf.name_scope('cost'):
    #             self.cost = self.outOfLarynxCycleCost
    #             self.variableSummaries(self.cost)
    #             pass
    #         pass
    #     pass


    # def evaluator(self):
    #     with tf.name_scope('Evaluator'):
    #         self.nGCIs = tf.cast(tf.reduce_sum(self.gciCount), dtype=tf.float32)
    #         with tf.name_scope('wave_Evaluator'):
    #             self.wave_correct_pred = tf.equal(tf.argmin(self.logits, 2), tf.argmin(self.y, 2))
    #             self.wave_accuracy = tf.reduce_mean(tf.cast(self.wave_correct_pred, tf.float32))
    #             self.variableSummaries(self.wave_accuracy)
    #         with tf.name_scope('gci_Evaluator'):
    #             self.gci_correct_pred = tf.equal(tf.argmin(self.maskedLogits), tf.argmin(self.maskedLabels))
    #             self.gci_accuracy = tf.reduce_mean(tf.cast(self.gci_correct_pred, tf.float32))
    #             self.variableSummaries(self.gci_accuracy)
    #         pass
    #     pass

    def evaluator(self):
        with tf.name_scope('Evaluator'):
            difference = tf.cast(tf.subtract(tf.argmin(self.logits, 2), tf.argmin(self.y, 2)), tf.float32)
            diffShape = tf.shape(difference)  # (BatchSize, timeSteps)
            difference = tf.reshape(difference, shape=[diffShape[0], diffShape[1], 1])  # (BatchSize, timeSteps, 1)
            self.markedResult = tf.matmul(self.maskMatrix, difference)  # Shape: (BatchSIze, timeSteps, 1)
            #
            self.nonzero = tf.cast(tf.count_nonzero(self.markedResult), dtype=tf.float32)
            falseAlarm = tf.cast(tf.count_nonzero(tf.nn.relu(self.markedResult)), dtype=tf.float32)
            self.nGCIs = tf.cast(tf.reduce_sum(self.gciCount), dtype=tf.float32)
            correct = self.nonzero
            miss = self.nGCIs - correct - falseAlarm
            # falseAlarm = tf.subtract(nonzero, miss)
            with tf.name_scope('miss'):
                self.missRate = tf.div(miss, self.nGCIs)
                self.variableSummaries(self.missRate)
                pass
            with tf.name_scope('correctRate'):
                self.correctRate = tf.div(correct, self.nGCIs)
                self.variableSummaries(self.correctRate)
                pass
            with tf.name_scope('falseAlarmed'):
                self.falseAlarmedRate = tf.div(falseAlarm, self.nGCIs)
                self.variableSummaries(self.falseAlarmedRate)
                pass
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
        #  Training epoch
        self.epoch = 0
        # Start a session and run up.
        with tf.Session(config=self.config) as sess:
            logging.info("Training session started!")
            sess.run(tf.global_variables_initializer())
            '''Restore model.'''
            if self.modelRestorePath is not None:
                self.saver.restore(sess, self.modelRestorePath)
                logging.info("Model restored from:" + str(self.modelRestorePath))
                pass
            # Summary
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.summarySavePath, sess.graph)
            # Prepare data set.
            self.openTrainingDataFile()
            trainingDataSet = InputReader(self.trainingDataFile, self.batchSize, self.timeStepSize)
            for i in range(self.trainIteration):
                (batchX, batchY, batchMask, batchGCICount) = trainingDataSet.getBatch(i)
                summary, _, trainingCost, trainingLarynxCycleCost, trainingOutOfLarynxCycleCost, \
                trainingNGCIs, estimatedGCICount = sess.run(
                    [merged, self.update, self.cost, self.larynxCycleCost, self.outOfLarynxCycleCost,
                     self.nGCIs, self.estimatedGCICount],
                    feed_dict={self.x: batchX, self.y: batchY, self.maskMatrix: batchMask,
                               self.gciCount: batchGCICount, self.keep_prob: 1.0})
                logging.info("Iteration:" + str(i)
                             + ", \tbatch loss= {:.9f}".format(trainingCost)
                             + ", \tlarynx loss= {:.9f}".format(trainingLarynxCycleCost)
                             + ", \tout of larynx loss= {:.9f}".format(trainingOutOfLarynxCycleCost)
                             + ", \tnGCIs= {:.9f}".format(trainingNGCIs)
                             + ", \testimated nGCIs= {:.9f}".format(estimatedGCICount)
                             # + ", \tmiss and false alarmed= {:.9f}".format(missAndFalseCount)
                             # + ", \tmissAndFalseCount = {:.9f}".format(missAndFalseCount)
                             )
                # logging.debug("batchX:" + str(batchX[0]))
                # logging.debug("batchY:" + str(batchY[0]))
                #
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
                    trainCost, trainingLarynxCycleCost, trainingOutOfLarynxCycleCost, \
                    trainingNGCIs, estimatedGCICount, trainingResults = sess.run(
                        [self.cost, self.larynxCycleCost, self.outOfLarynxCycleCost,
                         self.nGCIs, self.estimatedGCICount, self.results],
                        feed_dict={self.x: batchX, self.y: batchY, self.maskMatrix: batchMask,
                                   self.gciCount: batchGCICount, self.keep_prob: 1.0})
                    logging.info("Epoch:" + str(trainingDataSet.completedEpoch)
                                 + ", \titeration:" + str(i)
                                 + ", \tbatch loss= {:.9f}".format(trainCost)
                                 + ", \tlarynx loss= {:.9f}".format(trainingLarynxCycleCost)
                                 + ", \tout of larynx loss= {:.9f}".format(trainingOutOfLarynxCycleCost)
                                 + ", \tnGCIs= {:.9f}".format(trainingNGCIs)
                                 + ", \testimated nGCIs= {:.9f}".format(estimatedGCICount)
                                 # + ", \t training correct rate= {:.9f}".format(trainCorrectRate)
                                 # + ", \t training miss rate= {:.9f}".format(trainMissRate)
                                 # + ", \t training false alarmed rate= {:.9f}".format(trainFalseAlarmedRate)
                                 )
                    '''Print location example'''
                    # resultWriter = ResultWriter(samplingRate, frameSize=17, frameStride=9)
                    y = batchY[0]
                    targets = trans2DLabelSeq2Locations(y, self.samplingRate, self.frameSize, self.frameStride)
                    logging.info("trainingTargets:" + str(targets))
                    labelSeq = trainingResults[0]
                    locations = trans1DLabelSeq2Locations(labelSeq, self.samplingRate, self.frameSize, self.frameStride)
                    logging.info("trainingResults:" + str(locations))
                    pass
                # Validate
                if (trainingDataSet.completedEpoch > self.epoch):
                    validationDataSet = InputReader(self.validationDataFile, batchSize=1, maxTimeStep=self.timeStepSize)
                    validationIteration = validationDataSet.getBatchCount()
                    logging.info("Total validation iteration:" + str(validationIteration))
                    validationCostSum = 0.0
                    validationMissRateSum = 0.0
                    validationCorrectRateSum = 0.0
                    validationFalseAlarmedRateSum = 0.0
                    for i in range(validationIteration):
                        (batchX, batchY, batchMask, batchGCICount) = validationDataSet.getBatch(i)
                        validationCost, validationLarynxCycleCost, validationOutOfLarynxCycleCost, validationResults, validationNGCIs, estimatedGCICount = \
                            sess.run([self.cost, self.larynxCycleCost, self.outOfLarynxCycleCost, self.results, self.nGCIs, self.estimatedGCICount], feed_dict={self.x: batchX, self.y: batchY, self.maskMatrix: batchMask,
                                       self.gciCount: batchGCICount,
                                       self.keep_prob: 1.0})
                        logging.info("Validation epoch:" + str(validationDataSet.completedEpoch)
                                     + ", \tValidation iteration:" + str(i)
                                     + ", \tloss= {:.9f}".format(validationCost)
                                     + ", \tlarynx loss= {:.9f}".format(validationLarynxCycleCost)
                                     + ", \tout of larynx loss= {:.9f}".format(validationOutOfLarynxCycleCost)
                                     + ", \tnGCIs= {:.9f}".format(validationNGCIs)
                                     + ", \testimated nGCIs= {:.9f}".format(estimatedGCICount)
                                     # + ", \tcorrect rate= {:.9f}".format(validationCorrectRate)
                                     # + ", \tmiss rate= {:.9f}".format(validationMissRate)
                                     # + ", \tfalse alarmed rate= {:.9f}".format(validationFalseAlarmedRate)
                                     )
                        '''Print location example'''
                        # resultWriter = ResultWriter(samplingRate, frameSize=17, frameStride=9)
                        y = batchY[0]
                        targets = trans2DLabelSeq2Locations(y, self.samplingRate, self.frameSize, self.frameStride)
                        logging.info("validationTargets:" + str(targets))
                        labelSeq = validationResults[0]
                        locations = trans1DLabelSeq2Locations(labelSeq, self.samplingRate, self.frameSize, self.frameStride)
                        logging.info("validationResults:" + str(locations))
                        validationCostSum += validationCost
                    #     validationMissRateSum += validationMissRate
                    #     validationCorrectRateSum += validationCorrectRate
                    #     validationFalseAlarmedRateSum += validationFalseAlarmedRate
                        pass

                    logging.info("Epoch:" + str(validationDataSet.completedEpoch)
                                 + ", \titeration:" + str(i)
                                 + ", \tmean loss= {:.9f}".format(validationCostSum / validationIteration)
                    #              + ", \tmean validation correct rate= {:.9f}".format(
                    #     validationCorrectRateSum / validationIteration)
                    #              + ", \tmean validation miss rate= {:.9f}".format(
                    #     validationMissRateSum / validationIteration)
                    #              + ", \tmean validation false alarmed rate= {:.9f}".format(
                    #     validationFalseAlarmedRateSum / validationIteration)
                                 )
                    # Update epoch.
                    self.epoch = trainingDataSet.completedEpoch
                    pass
                    logging.info("Validation Finished!")
                    self.globalStep += 1
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
                logging.info("Model restored from:" + str(self.modelRestorePath))
            else:
                logging.info("Model restore failed.")
                return
            '''Prepare testing parameters.'''
            self.openTestingDataFile()
            # Prepare data set.
            testDataSet = InputReader(dataFile=self.testingDataFile, batchSize=1, maxTimeStep=self.timeStepSize)
            # Test iteration.
            testIteration = testDataSet.getBatchCount()
            logging.info("Total testing iteration:" + str(testIteration))
            logging.info("Testing starting!")
            '''Forward testing data.'''
            for i in range(testIteration):
                (batchX, batchY, batchMask, batchGCICount) = testDataSet.getBatch(i)
                testResults, testNGCIs, estimatedGCICount, testingCost = sess.run([self.results, self.nGCIs, self.estimatedGCICount, self.cost],
                                                                   feed_dict={self.x: batchX, self.y: batchY,
                                                                              self.maskMatrix: batchMask,
                                                                              self.gciCount: batchGCICount,
                                                                              self.keep_prob: 1.0})
                logging.info(", \tTesting iteration:" + str(i)
                             + ", \tloss= {:.9f}".format(testingCost)
                             + ", \tnGCIs= {:.9f}".format(testNGCIs)
                             + ", \testimated nGCIs= {:.9f}".format(estimatedGCICount)
                             )
                y = batchY[0]
                targets = trans2DLabelSeq2Locations(y, self.samplingRate, self.frameSize, self.frameStride)
                logging.info("testTargets:" + str(targets))
                labelSeq = testResults[0]
                locations = trans1DLabelSeq2Locations(labelSeq, self.samplingRate, self.frameSize, self.frameStride)
                logging.info("testResults:" + str(locations))
                '''Save output'''
                keyList = testDataSet.getBatchKeyList(i)
                self.resultWriter.saveBatchResult(testResults, keyList)
                pass
            pass
            self.closeTestingDataFile()
        logging.info("Testing finished!")
        pass

    pass
