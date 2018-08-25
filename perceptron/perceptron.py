
# coding: utf-8

# In[20]:


# coding: utf-8
#感知器实现与操作（或操作），只有一层隐藏层，即输出层。

import tensorflow as tf
from numpy.random import RandomState 

# 定义训练数据的大小
batch_size = 8

# 创建变量
w = tf.Variable(tf.truncated_normal([2,1],stddev=1,seed=1),name='weight')
b = tf.Variable(tf.truncated_normal([1],stddev=1,seed=1),name='bias')

# 设置占位符
x = tf.placeholder(tf.float32,shape=(None,2),name='input')
y_ = tf.placeholder(tf.float32, shape=(None,1),name='output')

# 感知机模型（原始形式）
a = tf.matmul(x,w)+b
y = tf.sigmoid(a)

# 损失函数
# 均方误差
cross_entropy = tf.reduce_mean(tf.square(y - y_))

# 学习率
learning_rate = 0.01

# 感知机学习算法（采用梯度下降法），优化参数
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# 随机产生0/1训练数据集
rmd = RandomState(1)
dataset_size = 128
X = rmd.randint(2,size=(dataset_size,2))

# 定义标签规则，在这里所有的x1&x2的结果作为Y值，0表示负样本，1表示正样本
# 与操作
#Y = [[int(x1) and int(x2)] for (x1,x2) in X]
# 或操作
Y = [[int(x1) or int(x2)] for (x1,x2) in X]

# 创建会话
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(w))
    print(sess.run(b))
    
    steps = 5000
    for i in range(steps):
        #每次选取batch_size个样本数据进行训练
        start = (i*batch_size)%dataset_size
        end = min(start+batch_size, dataset_size)
        #train_x = [[1.0,1.0],[0.0,0.0],[1.0,0.0],[0.0,1.0]]
        #train_y = [[1.0],[0.0],[0.0],[0.0]]
        #sess.run(train_step, feed_dict={x:train_x,y_:train_y})
        sess.run(train_step, feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%100 == 0:
            #total_cross_entropy = sess.run(cross_entropy,feed_dict={x:train_x,y_:train_y})
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training step(s),cross_entropy on all data is %g"%(i, total_cross_entropy))
    print(sess.run(w))
    print(sess.run(b))
    test_x = [[0.0,1.0],[0.0,0.0],[1.0,1.0],[1.0,0.0]]
    print(sess.run(y, feed_dict={x:test_x}))

