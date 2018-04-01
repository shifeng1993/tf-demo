import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 设置tensorflow对GPU的使用按需分配
config = tf.ConfigProto(allow_soft_placement=True)

# 定义数据集
mnist = input_data.read_data_sets(
    './MNIST-data', one_hot=True)  # 将外部数据读取到data里面


# 权值初始化
def weight_variable(shape):
    # 用正态分布来初始化权值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 本例中用relu激活函数，所以用一个很小的正偏置较好
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积层
def conv2d(x, W):
    # 默认 strides[0]=strides[3]=1, strides[1]为x方向步长，strides[2]为y方向步长
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# pooling 层
def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


X_ = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 把X转为卷积所需要的形式
X = tf.reshape(X_, [-1, 28, 28, 1])
# 第一层卷积：5×5×1卷积核32个 [5，5，1，32],h_conv1.shape=[-1, 28, 28, 32]
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)

# 第一个pooling 层[-1, 28, 28, 32]->[-1, 14, 14, 32]
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积：5×5×32卷积核64个 [5，5，32，64],h_conv2.shape=[-1, 14, 14, 64]
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 第二个pooling 层,[-1, 14, 14, 64]->[-1, 7, 7, 64]
h_pool2 = max_pool_2x2(h_conv2)

# flatten层，[-1, 7, 7, 64]->[-1, 7*7*64],即每个样本得到一个7*7*64维的样本
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# fc1
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 1.损失函数：cross_entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# 2.优化函数：AdamOptimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 3.预测准确结果统计
#　预测值中最大值（１）即分类结果，是否等于原始标签中的（１）的位置。argmax()取最大值所在的下标
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 如果一次性来做测试的话，可能占用的显存会比较多，所以测试的时候也可以设置较小的batch来看准确率
test_acc_sum = tf.Variable(0.0)
batch_acc = tf.placeholder(tf.float32)
new_test_acc_sum = tf.add(test_acc_sum, batch_acc)
update = tf.assign(test_acc_sum, new_test_acc_sum)

# 启动图 (graph)
with tf.Session(config=config) as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 训练
    for i in range(2000):
        X_batch, y_batch = mnist.train.next_batch(batch_size=50)
        if i % 500 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                X_: X_batch,
                y_: y_batch,
                keep_prob: 1.0
            })
            print('step:', i, '   ', 'training acc:', train_accuracy)
        train_step.run(feed_dict={X_: X_batch, y_: y_batch, keep_prob: 0.5})

    # 全部训练完了再做测试，batch_size=100
    for i in range(100):
        X_batch, y_batch = mnist.test.next_batch(batch_size=100)
        test_acc = accuracy.eval(feed_dict={
            X_: X_batch,
            y_: y_batch,
            keep_prob: 1.0
        })
        update.eval(feed_dict={batch_acc: test_acc})
        if (i + 1) % 20 == 0:
            print('testing step:', i + 1, '   ', 'test acc:',
                  test_acc_sum.eval())
    print(" test_accuracy ", '%', test_acc_sum.eval() / 100.0)
