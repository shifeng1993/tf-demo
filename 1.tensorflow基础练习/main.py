import numpy as np
import tensorflow as tf

x = tf.constant([[1, 2]], name='x', dtype='int32')
y = tf.constant([[3],[4]], name='y', dtype='int32')

# op1 = tf.add(3,  6)                                          # 3 + 6 = 9
# op2 = tf.multiply(3,  6)                                     # 3 * 6 = 18
total_op = tf.matmul(x, y)                             # 9 * 18 = 162 
xy_op = tf.multiply(x, y)
with tf.Session() as sess:
  log = tf.summary.FileWriter('./graphs', sess.graph)       # 输出数据流图到graphs文件夹下 可以用“tensorboard --logdir=./graphs” 查看
  log.close()
  print(sess.run(total_op))
  print(sess.run(xy_op))
  