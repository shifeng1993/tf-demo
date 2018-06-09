import tensorflow as tf
import numpy as np

# 设置提示信息的等级 menu['1','2','3'] 默认'1'显示所有信息,'2'只显示 warning 和 Error,'3'只显示 Error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1.准备数据：使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.random.rand(2, 100)  # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 2.构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, tf.cast(x_data, tf.float32)) + b

# 3.求解模型
# 设置损失函数：误差的均方差
loss = tf.reduce_mean(tf.square(y - y_data))
# 选择梯度下降的方法
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 迭代的目标：最小化损失函数
train = optimizer.minimize(loss)

############################################################
# 以下是用 tf 来解决上面的任务
# 1.初始化变量：tf 的必备步骤，主要声明了变量，就必须初始化才能用
init = tf.global_variables_initializer()

# 设置tensorflow对GPU的使用按需分配
config = tf.ConfigProto(allow_soft_placement=True)
# 2.启动图 (graph)
sess = tf.Session(config=config)
sess.run(init)

# 3.迭代，反复执行上面的最小化损失函数这一操作（train op）,拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(str(step), sess.run(W), sess.run(b))
print('finished~~')
print('权重:', sess.run(W), '偏置:', sess.run(b))
# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]

# 需要注意的是np默认随机是float64，或者先用np.float32()直接初始化成float32，或者在引入计算时用tf.cast(xxx, tf.float32) 进行强制类型转换
