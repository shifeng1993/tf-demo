import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MNIST = input_data.read_data_sets('data', one_hot=True)                           # 将外部数据读取到data里面

question = tf.placeholder(tf.float32, [None, 784], name='question')               # 先占位，后面使用再填充
anwser = tf.placeholder(tf.float32, [None, 10], name='anwser')                    # 先占位，后面使用再填充

# 第一层神经网络
weight_layer1 = tf.Variable(tf.random_normal([784, 500]), name='weight_layer1')   # 第一层神经网络权重
biases_layer1 = tf.Variable(tf.random_normal([500]), name='biases_layer1')        # 第一层神经网络偏置量
# 第二层神经网络
weight_layer2 = tf.Variable(tf.random_normal([500, 10]), name='weight_layer2')    # 第二层神经网络权重
biases_layer2 = tf.Variable(tf.random_normal([10]), name='biases_layer2')         # 第二层神经网络偏置量


def rabbit(the_question):
  layer1 = tf.matmul(the_question, weight_layer1) + biases_layer1                 # 第一层神经网络，线性函数 y = wx + b
  return tf.matmul(layer1, weight_layer2) + biases_layer2                         # 第二层神经网络，线性函数 y = wx + b

def wolf(anwser_of_rabbit, the_answer):
  badly = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=anwser_of_rabbit, labels=tf.argmax(the_answer, 1)) # arg_max稀疏阵转化为密度阵
  return tf.train.GradientDescentOptimizer(0.1).minimize(badly)                   # 梯度下降x间隔，找到梯度函数的极小值


def grade():                                                                      # 测试集
  anwser_of_rabbit = rabbit(question)                                             # 输出第一个实体的y值
  grades = tf.equal(tf.argmax(anwser_of_rabbit, 1), tf.argmax(anwser, 1))         # 对比正确答案
  return tf.reduce_mean(tf.cast(grades, tf.float32))                              # 计算张量维度的平均值  tf.cast转化成float值

circulation = wolf(rabbit(question), anwser)                                      # 先实例化，在循环中可以减少开销
G = grade()                                                                       # 先实例化，在循环中可以减少开销
saver = tf.train.Saver()                                                          # 先实例化，在循环中可以减少开销

with tf.Session() as sess:
  for i in range(1000):
    tf.global_variables_initializer().run()                                        # 初始化内存中的variables
    if i%100 == 0: print(i)
    x, y = MNIST.train.next_batch(20)   
    sess.run(circulation, feed_dict={question:x,anwser:y})                         # 拿出20个数组填充到x和y里面
  x0 = MNIST.validation.images                                                     # 定义测试集
  y0 = MNIST.validation.labels                                                     # 定义测试集
  print(sess.run(G,feed_dict={question:x0, anwser: y0}) * 100)                     # 输出测试结果
  saver.save(sess, 'models/model.ckpt', global_step=1000)                          # 进行训练并保存到本地models文件夹下model文件 ，globals_step 训练 次数
