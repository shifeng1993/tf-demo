# 引入库
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1.数据准备，pandas ，scikit-learn 自己选择进行数据加载到内存中
plt.rcParams['figure.figsize'] = (14, 8)                            # 可视化时图的长和宽

n_ob = 100                                                          # 样本数，定义100个

xs = np.linspace(-3, 3, n_ob)                                       # x范围内的点
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_ob)                # y范围的sin点 加随机扰动
# plt.scatter(xs, ys)                                                 # 散点图绘制

# 2.准备好placeholder
x = tf.placeholder(tf.float32, name="x")
y = tf.placeholder(tf.float32, name="y")

# 3.初始化参数/权重
w = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# 4.计算预测结果
y_pred = tf.add(tf.multiply(w, x), b)                               # y = wx + b

# 5.计算损失函数
loss = tf.square(y - y_pred, name="loss")                           # 平方损失函数 loss2

# 6.初始化op， 如果是梯度下降，可以选择Optimizer，神经网络选用的话会陷入局部最低。可以用别的
learning_rate = 0.01                                                # 梯度下降步长
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 7.指定迭代次数，执行计算
n_samples = xs.shape[0]
init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)                                                    # 先初始化全局变量
  log = tf.summary.FileWriter("./graphs", sess.graph)               # 输出图文件，方便在tensorbload里面查看
  
  # 训练模型
  total_loss = 0
  for i in range(100):
    for m, n in zip(xs, ys):                                        # zip函数可以得到两个参数组成的元祖
      l = sess.run([optimizer, loss], feed_dict={x:m,y:n})
      total_loss += l[1][0]                                         # 每个迭代的损失都加到总的损失上
    if i%20 == 0:                                                   # 每隔20次
      print("epoch {0}:{1}".format(i, total_loss/n_samples))        # 输出一次当前的平方损失，loss2
  
  # 关闭log输出
  log.close()

  # 取出 w 和 b 的值 打印
  weight = w.eval()                                                 # 取出变量 w 的值
  bias = b.eval()                                                   # 取出变量 b 的值
  print("weight:{0}".format(weight[0]))                             # 打印出线性函数的权重
  print("bias:{0}".format(bias[0]))                                 # 打印出线性函数的偏置
  plt.plot(xs, ys, 'bo', label="sample")                            # 描绘散点
  plt.plot(xs, xs* weight + bias, 'r', label="linear")              # 描绘线性函数
  plt.legend()                                                      #
  plt.show()                                                        # 展示
