import tensorflow.contrib.rnn as rnn


class LSTMModel(DNNModel):
    def __DNNModel(self):
        with tf.name_scope('DNNModel'):
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
            self.outputs, _ = tf.nn.dynamic_rnn(self.stack, self.x, dtype=tf.float32)
            self.logits = tf.contrib.layers.fully_connected(self.outputs, self.classNum, activation_fn=None)
            self.variableSummaries(self.logits)
            pass
        pass

    def lossFunction(self):
        with tf.name_scope('LossFunction'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
            self.cost = tf.reduce_mean(cross_entropy)
            # self.cost = tf.reduce_mean(ctc_ops.ctc_loss(labels=self.y, inputs=self.logits, sequence_length=self.timestepSize, time_major=False))
            self.variableSummaries(self.cost)
            pass
        pass

    pass
