import tensorflow as tf

# 原始列表
arr = tf.constant([1, 2, 3, 4, 5])

# 给一个初始值 每次叠加完成就会替换
total_sum = tf.Variable(0, dtype=tf.float32, name='sum')

# 每次需要加的值，先占位
every_add = tf.placeholder(tf.float32)

# 叠加产生的新值，用来更新旧值
total_sum_new = tf.add(total_sum, every_add)

# 更新函数
update = tf.assign(total_sum, total_sum_new)

# 求平均


# 初始化所有变量
init = tf.global_variables_initializer()

# 开启计算
with tf.Session() as sess:
    sess.run(init)                                                  # 运行初始化
    arr_len = len(sess.run(arr))                                    # 获取迭代次数
    for i in range(arr_len):                                        # 进行迭代
        sess.run(update, feed_dict={every_add: sess.run(arr[i])})   # 运行更新函数，会自动找到依赖的图，进行计算
    print(sess.run(arr))
    print(sess.run(total_sum))
    print(sess.run(total_sum)/arr_len)                              # 输出迭代完成的总和，完成平均数的输出
