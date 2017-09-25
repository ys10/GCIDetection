from __future__ import print_function

import logging
import tensorflow as tf
import h5py
from tensorflow.contrib import rnn
from src.DataProcess import Dataset

# 设置 GPU 按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

hdf5DirPath = "../../data/hdf5/"
hdf5Filename = "APLAWDW_s_01_a"
hdf5Extension = ".hdf5"

dataFile = h5py.File(hdf5DirPath + hdf5Filename + hdf5Extension)

# 学习率
lr = 1e-3
# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
# batchSize = tf.placeholder(tf.int32)  # 注意类型必须为 tf.int32
batchSize = 16
# batch_size = 128

# 每个时刻的输入特征是40000维的，就是每个时刻输入一行，一行有 1 个像素
inputSize = 1
# 时序持续长度为40000，即每做一次预测，需要先输入40000行
timestepSize = 40000
# 每个隐含层的节点数
hiddenSize = 256
# LSTM layer 的层数
layerNum = 1
# 最后输出分类类别数量，如果是回归预测的话应该是 1
classNum = 2

X = tf.placeholder(tf.double, [None, inputSize * timestepSize])
y = tf.placeholder(tf.bool, [None, classNum])
keep_prob = tf.placeholder(tf.double)

# 把784个点的字符信息还原成 28 * 28 的图片
# 下面几个步骤是实现 RNN / LSTM 的关键
####################################################################
# **步骤1：RNN 的输入shape = (batch_size, timestep_size, input_size)
X = tf.reshape(X, [-1, timestepSize, inputSize])

# **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
lstmCell = rnn.BasicLSTMCell(num_units=hiddenSize, forget_bias=1.0, state_is_tuple=True)

# **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
lstmCell = rnn.DropoutWrapper(cell=lstmCell, input_keep_prob=1.0, output_keep_prob=keep_prob)

# **步骤4：调用 MultiRNNCell 来实现多层 LSTM
mlstmCell = rnn.MultiRNNCell([lstmCell] * layerNum, state_is_tuple=True)

# **步骤5：用全零来初始化state
initState = mlstmCell.zero_state(batchSize, dtype=tf.double)

# **步骤6：按时间步展开计算
outputs = list()
state = initState
with tf.variable_scope('RNN'):
    for timestep in range(timestepSize):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        # 这里的state保存了每一层 LSTM 的状态
        (cell_output, state) = mlstmCell(X[:, timestep], state)
        outputs.append(cell_output)
h_state = outputs[-1]

# 开始训练和测试
W = tf.Variable(tf.truncated_normal([hiddenSize, classNum], stddev=0.1), dtype=tf.double)
bias = tf.Variable(tf.constant(0.1,shape=[classNum]), dtype=tf.double)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)


# 损失和评估函数
cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "double"))

sess.run(tf.global_variables_initializer())

dataSet = Dataset.Data(dataFile, batchSize, timestepSize)
for i in range(50):
    # _batch_size = 4
    [batchX, batchY] = dataSet.getBatch(i)
    if (i+1)% 5 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={X:batchX, y: batchY, keep_prob: 1.0})
        # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
        logging.info("Epoch:%d,\t iteration:%d,\t training accuracy:%g" % ( dataSet.completedEpoch, (i+1), train_accuracy))
    sess.run(train_op, feed_dict={X:batchX, y: batchY, keep_prob: 1.0})
