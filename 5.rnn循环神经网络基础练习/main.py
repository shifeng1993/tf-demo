import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

# 1.数据准备
mnist = input_data.read_data_sets('./data', one_hot=True)

# 2.全局参数配置
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# 3.网络参数
n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

# 4.准备好placeholder
x = tf.placeholder('float', [None, n_steps, n_input], name='X_placeholder')
y = tf.placeholder('float', [None, n_classes], name='Y_placeholder')

# 5.准备好权重和偏置
weight = {
  'out': tf.Variable(tf.random_normal([n_hidden, n_classes]), name='weight')
}
bias = {
  'out': tf.Variable(tf.random_normal([n_classes]), name='bias')
}

# 6.定义rnn
def RNN(x, weight, bias):
  x = tf.unstack(x, n_steps, 1)
  lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
  outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
  return tf.matmul(outputs[-1], weight['out']) + bias['out']

# 7.拿到预测结果
pred = RNN(x, weight, bias)

# 8.计算损失并指向op
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y),name='cost')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 9.评估模型
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# 10.初始化全部变量
init = tf.global_variables_initializer()

# 11.运行计算
with tf.Session() as sess:
  sess.run(init)
  log = tf.summary.FileWriter('./graphs', sess.graph)
  step = 1
  while step * batch_size < training_iters:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size,n_steps,n_input))
    sess.run(optimizer, feed_dict={x:batch_x,y:batch_y})
    if step % display_step == 0:
      acc = sess.run(accuracy, feed_dict={x:batch_x,y:batch_y})
      loss = sess.run(cost, feed_dict={x:batch_x,y:batch_y})
      print('iter ' + str(step*batch_size) + ': minibatch loss = ' + str(loss) + ', training accuracy = ' + str(acc))
    step += 1
  print('finish')
  log.close()
  
  # 测试模型 输出精度
  test_len = 128
  test_data = mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
  test_label = mnist.test.labels[:test_len]
  print('testing accuracy', sess.run(accuracy,feed_dict={x:test_data,y:test_label}))