from dnnModel.DNNModel import *
import tensorflow.contrib.rnn as rnn


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
            linearOutput = tf.nn.softmax(
                tf.contrib.layers.fully_connected(rnnOutputs, self.outputSize, activation_fn=None))
            with tf.name_scope('logits'):
                self.logits = linearOutput
                self.variableSummaries(self.logits)
                pass
            # self.cost = tf.reduce_sum(tf.square(tf.cast(self.logits, dtype=tf.float32) - self.y))
            pass
        pass

    def lossFunction(self):
        with tf.name_scope('LossFunction'):
            with tf.name_scope('difference'):
                self.difference = tf.subtract(self.logits, self.y)
            with tf.name_scope('costDistribute'):
                self.costDistribute = tf.matmul(self.mask, self.difference)
                # self.costDistribute = tf.boolean_mask( self.difference, self.mask)
            with tf.name_scope('cost'):
                self.cost = tf.reduce_mean(tf.square(self.costDistribute))
                self.variableSummaries(self.cost)
                pass
            pass
        pass

    def evaluator(self):
        with tf.name_scope('Evaluator'):
            difference = tf.cast(tf.subtract(tf.argmin(self.logits, 2), tf.argmin(self.y, 2)), tf.float32)
            diffShape = tf.shape(difference)  # (BatchSize, timeSteps)
            difference = tf.reshape(difference, shape=[diffShape[0], diffShape[1], 1])  # (BatchSize, timeSteps, 1)
            markedResult = tf.matmul(self.mask, difference)  # Shape: (BatchSIze, timeSteps, 1)
            #
            nonzero = tf.cast(tf.count_nonzero(markedResult), dtype=tf.float32)
            falseAlarm = tf.cast(tf.count_nonzero(tf.nn.relu(markedResult)), dtype=tf.float32)
            nGCIS = tf.reduce_sum(self.gciCount)
            correct = nGCIS - nonzero
            miss = nonzero - falseAlarm
            # falseAlarm = tf.subtract(nonzero, miss)
            with tf.name_scope('miss'):
                self.missRate = tf.div(miss, nGCIS)
                self.variableSummaries(self.missRate)
                pass
            with tf.name_scope('correctRate'):
                self.correctRate = tf.div(correct, nGCIS)
                self.variableSummaries(self.correctRate)
                pass
            with tf.name_scope('falseAlarmed'):
                self.falseAlarmedRate = tf.div(falseAlarm, nGCIS)
                self.variableSummaries(self.falseAlarmedRate)
                pass
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
                summary, _, trainingCost = sess.run([merged, self.optimizer, self.cost],
                                                    feed_dict={self.x: batchX, self.y: batchY, self.mask: batchMask,
                                                               self.gciCount: batchGCICount, self.keep_prob: 1.0})
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
                    trainCost, trainMissRate, trainCorrectRate, trainFalseAlarmedRate = sess.run(
                        [self.cost, self.missRate, self.correctRate, self.falseAlarmedRate],
                        feed_dict={self.x: batchX, self.y: batchY, self.mask: batchMask, self.gciCount: batchGCICount,
                                   self.keep_prob: 1.0})
                    logging.info("Epoch:" + str(trainingDataSet.completedEpoch)
                                 + ", \titeration:" + str(i)
                                 + ", \tbatch loss= {:.6f}".format(trainCost)
                                 + ", \t training correct rate= {:.6f}".format(trainCorrectRate)
                                 + ", \t training miss rate= {:.6f}".format(trainMissRate)
                                 + ", \t training false alarmed rate= {:.6f}".format(trainFalseAlarmedRate)
                                 )
                    pass
                # Validate
                if (trainingDataSet.completedEpoch > self.epoch):
                    validationDataSet = InputReader(self.validationDataFile, batchSize=1, maxTimeStep=self.timeStepSize)
                    validationIteration = validationDataSet.getBatchCount()
                    validationCostSum  = 0.0
                    validationMissRateSum  = 0.0
                    validationCorrectRateSum  = 0.0
                    validationFalseAlarmedRateSum = 0.0
                    for i in range(validationIteration):
                        validationCost, validationMissRate, validationCorrectRate, validationFalseAlarmedRate = sess.run(
                            [self.cost, self.missRate, self.correctRate, self.falseAlarmedRate],
                            feed_dict={self.x: batchX, self.y: batchY, self.mask: batchMask,
                                       self.gciCount: batchGCICount,
                                       self.keep_prob: 1.0})
                        validationCostSum += validationCost
                        validationMissRateSum += validationMissRate
                        validationCorrectRateSum += validationCorrectRate
                        validationFalseAlarmedRateSum +=validationFalseAlarmedRate
                        pass
                    logging.info("Epoch:" + str(validationDataSet.completedEpoch)
                                 + ", \titeration:" + str(i)
                                 + ", \tmean loss= {:.6f}".format(validationCostSum / validationIteration)
                                 + ", \tmean validation correct rate= {:.6f}".format(validationCorrectRateSum / validationIteration)
                                 + ", \tmean validation miss rate= {:.6f}".format(validationMissRateSum / validationIteration)
                                 + ", \tmean validation false alarmed rate= {:.6f}".format(validationFalseAlarmedRateSum / validationIteration)
                                 )
                    # Update epoch.
                    self.epoch = trainingDataSet.completedEpoch
                    pass
                    logging.info("Validation Finished!")
                pass
            self.closeTrainingDataFile()
            logging.info("Optimization Finished!")
            pass
        pass

    pass
