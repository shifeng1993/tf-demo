import tensorflow as tf
import numpy as np

# 设置提示信息的等级 menu['1','2','3'] 默认'1'显示所有信息,'2'只显示 warning 和 Error,'3'只显示 Error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input1 = tf.constant(1)
input2 = tf.constant(2)
input3 = tf.constant(3)

op1 = tf.add(input1, input2)
op2 = tf.multiply(op1, input3)

with tf.Session() as sess:
    print(sess.run([op1, op2]))   # 输出多个op应写成数组