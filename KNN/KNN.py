import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp/data/", one_hot=True)

#从数据集中获取5000个样本作为训练集，200个样本作为测试集
Xtrain, Ytrain = mnist.train.next_batch(5000)
Xtest, Ytest = mnist.test.next_batch(200)

#输入占位符
xtr = tf.placeholder("float",[None, 784])
xte = tf.placeholder("float",[784])

#使用曼哈顿距离，L1(xi,xj)=∑nl=1|x(l)i−x(l)j|
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)

#获取最小距离的训练样本的索引
pred = tf.argmin(distance, 0)

#分类精度
accuracy = 0

# 初始化变量
# init = tf.global_variables_initializer()
init = tf.initialize_all_variables()
#运行对话，训练模型
with tf.Session() as sess:
    sess.run(init)
    #遍历测试数据
    for i in range(len(Xtest)):
        #向占位符传入训练数据，获取当前样本的最近邻索引
        nn_index = sess.run(pred, feed_dict={xtr: Xtrain, xte: Xtest[i, :]})
        print("Test", i, "Prediction:", np.argmax(Ytrain[nn_index]), "True Class:", np.argmax(Ytest[i]))

        #计算精确度
        if np.argmax(Ytrain[nn_index]) == np.argmax(Ytest[i]):
            accuracy += 1./len(Xtest)
            print("accuracy", accuracy)
    print("Done")
    print("Accuracy:", accuracy)
