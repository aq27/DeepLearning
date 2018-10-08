__author__ = 'lenovo'

import tensorflow as tf
import numpy as np

#训练数据
num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

#定义参数
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="W")
b = tf.Variable(tf.zeros([1]), name="b")

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
y_pred = W * x + b

#定义损失函数
learning_rate = 0.01
loss = tf.reduce_mean(tf.square(y_pred - y), name="loss")
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#初始化参数
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for step in range(5000):
        sess.run(train_step, {x: x_data, y: y_data})
        print("W = ", sess.run(W), "b = ", sess.run(b), "loss = ", sess.run(loss, {x: x_data, y: y_data}))
    #测试
    x_test = [22.1, 33.5, 66.9, 12.3]
    for x in x_test:
        y_pred = x*W+b
        print("y_pred:", sess.run(y_pred), "y_act:", x*0.1+0.3)
writer = tf.summary.FileWriter("./tmp", sess.graph)




