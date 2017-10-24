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
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
            # cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=self.y, logits=self.logits, pos_weight=5000)
            with tf.name_scope('difference'):
                self.difference = tf.subtract(self.logits, self.y)
                # self.cost = tf.reduce_sum(tf.cast(self.difference, dtype=tf.float32))
            with tf.name_scope('costDistribute'):
                # maskShape = tf.shape(self.mask)  # (BatchSize, timeSteps, timeSteps)
                # tempMask = tf.reshape(self.mask, shape=[maskShape[0] * maskShape[1], maskShape[2]])
                # seqShape = tf.shape(self.difference)  # (BatchSize, timeSteps, outputSize)
                # tempSeq = tf.reshape(tf.transpose(self.difference, perm=[1, 0, 2]),
                #                      shape=[seqShape[1], seqShape[0] * seqShape[2]])
                # product = tf.matmul(tempMask, tempSeq)  # (BatchSize * timeSteps, BatchSize * outputSize)
                # self.costDistribute = tf.reshape(product, shape=[maskShape[0], maskShape[1], seqShape[2]])
                self.costDistribute = tf.matmul(self.mask, self.difference)
                # self.costDistribute = tf.transpose(product, perm=[1, 0, 2])
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
            # #
            # maskShape = tf.shape(self.mask)  # (BatchSize, timeSteps, timeSteps)
            # tempMask = tf.reshape(self.mask, shape=[maskShape[0] * maskShape[1], maskShape[2]])
            # seqShape = tf.shape(difference)  # (BatchSize, timeSteps, 1)
            # tempSeq = tf.reshape(tf.transpose(difference, perm=[1, 0, 2]),
            #                      shape=[seqShape[1], seqShape[0] * seqShape[2]])
            # product = tf.matmul(tempMask, tempSeq)  # (BatchSize * timeSteps, BatchSize * outputSize)
            # markedResult = tf.reshape(product, shape=[maskShape[0], maskShape[1], seqShape[2]])
            markedResult = tf.matmul(self.mask, difference)  # Shape: (BatchSIze, timeSteps, 1)
            #
            nonzero = tf.cast(tf.count_nonzero(markedResult), dtype=tf.float32)
            falseAlarm = tf.cast(tf.count_nonzero(tf.nn.relu(markedResult)), dtype=tf.float32)
            nGCIS = tf.reduce_sum(self.gciCount)
            # y, _, count = tf.unique_with_counts(tf.reshape(markedResult, shape=[-1]))
            # nGCIS = tf.cast(tf.reduce_sum(count), dtype=tf.float32)
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
            self.openDataFile()
            dataSet = InputReader(self.dataFile, self.batchSize, self.timeStepSize)
            for i in range(self.trainIteration):
                (batchX, batchY, batchMask, batchGCICount) = dataSet.getBatch(i)
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
                    logging.info("Epoch:" + str(dataSet.completedEpoch)
                                 + ", \titeration:" + str(i)
                                 + ", \tbatch loss= {:.6f}".format(trainCost)
                                 + ", \t training correct rate= {:.6f}".format(trainCorrectRate)
                                 + ", \t training miss rate= {:.6f}".format(trainMissRate)
                                 + ", \t training false alarmed rate= {:.6f}".format(trainFalseAlarmedRate)
                                 )
                    pass
                pass
            self.closeDataFile()
            logging.info("Optimization Finished!")
            pass
        pass

    pass
