import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# batch_size：每次训练使用的训练集的子集的大小
# image_width：图像的宽
# image_height：图像的高
# channels：彩色图像值为3，黑白图像值为1

# https://www.jianshu.com/p/96440832d7b9


def cnn_model_fn(features, labels, mode):
  # 输入层
  # 输入层的tensor（input_layer）一共有4个维度，分别是：[ batch_size, image_width, image_height, channels]
  # MNIST 图像是28x28像素, 一个颜色通道
  # features将大小为28×28的黑白照片通过feature参数传递进来（每个值代表一个像素），然后reshape成4-D的输入层tensor。batch_size取-1表示该值是在其他3个值不变的情况下，根据feature的长度动态计算出的。这样做的好处是可以把batch_size作为一个可以调的hyperparameter。
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # 卷积层 #1
  # 计算 32 个特征，使用5*5的卷积核，并使用relu激活函数来输出非线性结果
  # padding=‘same’：通过zero-padding的办法，保证input和output tensor的大小一致。
  # Input 张量形状: [batch_size, 28, 28, 1]
  # Output 张量形状: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # 池化层 #1
  # 2*2 最大池化，步长为2
  # Input 张量形状: [batch_size, 28, 28, 32]
  # Output 张量形状: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # 卷积层 #2
  # 计算64个特征，使用 5*5的卷积核，并使用relu激活函数来输出非线性结果
  # 添加内宽度 Padding 以保证宽高
  # Input 张量形状: [batch_size, 14, 14, 32]
  # Output 张量形状: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # 池化层 #2
  # 2*2 最大池化，步长为2
  # Input 张量形状: [batch_size, 14, 14, 64]
  # Output 张量形状: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # 将张量转化为向量
  # Input 张量形状: [batch_size, 7, 7, 64]
  # Output 张量形状: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # 全连接层 #1
  # 全连接层有 1024个神经元，使用ReLU激活函数
  # Input 张量形状: [batch_size, 7 * 7 * 64]
  # Output 张量形状: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # 添加 dropout 操作; dropout率0.4 (为了避免过拟合，在训练的时候，40%的神经元会被随机去掉）
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # 全连接层 #2（Logits Layer）10个神经元，每个神经元对应一个类别（0-9），返回预测的原始值
  # Input 张量形状: [batch_size, 1024]
  # Output 张量形状: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  # 定义字典 后面调用
  predictions = {
      # 生成的预测（预测和评估模式）
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # 计算损失
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # 配置训练的Op (用于训练模型)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # 添加测试标准 (测试模式)
  eval_metric_ops = {
      "accuracy":
      tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # 加载训练 和 测试数据
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # 创建一个评估测试
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="./mnist_convnet_model")

  # 建立预测日志
  # 用 "probabilities" 标签记录 “Softmax” 张量中的值
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # 训练这个模型
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn, steps=30000, hooks=[logging_hook])

  # 测试和打印结果
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
