import tensorflow as tf
import numpy as np

input1 = tf.constant(1)
input2 = tf.constant(2)
input3 = tf.constant(3)

op1 = tf.add(input1, input2)
op2 = tf.multiply(op1, input3)

with tf.Session() as sess:
    print(sess.run([op1, op2]))   # 输出多个op应写成数组