# import tensorflow as tf

# with tf.name_scope('hello1') as scope:
#   weights1 = tf.Variable([1.0, 2.0], name='weights')
#   bias1 = tf.Variable([0.3], name='bias')


# with tf.name_scope('hello2') as scope:
#   weights2 = tf.Variable([1.0, 2.0], name='weights')
#   bias2 = tf.Variable([0.3], name='bias')

# print(weights2.name)


import tensorflow as tf
with tf.variable_scope('v_scope') as scope1:
    Weights1 = tf.get_variable('Weights', shape=[2,3])
    bias1 = tf.get_variable('bias', shape=[3])

# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的变量必须已经定义过了，才能设置 reuse=True，否则会报错
with tf.variable_scope('v_scope', reuse=True) as scope2:
    Weights2 = tf.get_variable('Weights')

print(Weights1.name)
print(Weights2.name)