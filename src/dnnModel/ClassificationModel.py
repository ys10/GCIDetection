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
            self.outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                lstm_fw_cells,
                lstm_bw_cells,
                self.x,
                dtype=tf.float32
            )
            self.logits = tf.contrib.layers.fully_connected(self.outputs, self.classNum, activation_fn=None)
            self.variableSummaries(self.logits)
            pass
        pass

    def __lossFunction(self):
        with tf.name_scope('LossFunction'):
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
            cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=self.y, logits=tf.nn.softmax(self.logits), pos_weight=5000)
            self.cost = tf.reduce_mean(cross_entropy)
            self.variableSummaries(self.cost)
            pass
        pass

    pass