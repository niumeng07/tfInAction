#-*- coding:utf-8 -*-
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pdb

def xavier_init(fan_in, fan_out, constant = 1):
  low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
  high = constant * np.sqrt(6.0 / (fan_in + fan_out))
  return tf.random_uniform((fan_in, fan_out),
                         minval = low, maxval = high,
                         dtype = tf.float32)


def train(self, n_input):
  x = tf.placeholder(tf.float32, [None, n_input])

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

def standard_scale(X_train, X_test):
  preprocessor = prep.StandardScaler().fit(X_train)
  X_train = preprocessor.transform(X_train)
  X_test = preprocessor.transform(X_test)
  return X_train, X_test

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)


# 定义model
# 输入
x = tf.placeholder(tf.float32, [None, 784]) # [None, 784]样本条数, 样本特征维度
# 第一层参数
w1 = tf.Variable(xavier_init(784, 200))  # 200是当前隐层中神经元个数, 784是输入数
b1 = tf.Variable(xavier_init(1, 200))
# 第一层运算规则
hidden1 = tf.nn.softplus(tf.add(tf.matmul(x, w1), b1))

# 第二层
w2 = tf.Variable(xavier_init(200, 200))
b2 = tf.Variable(xavier_init(1, 200))
hidden2 = tf.nn.softplus(tf.add(tf.matmul(hidden1, w2), b2))

# 第三层参数
w3 = tf.Variable(xavier_init(200, 784))
b3 = tf.Variable(xavier_init(1, 784))
# 第三层去处规则，实际上就是输出层
final = tf.add(tf.matmul(hidden2, w3), b3)

# 损失函数
loss_fn = 0.5*tf.reduce_sum( tf.pow(tf.subtract(final, x ), 2.0) )
# 优化器
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
# 优化操作
optimize_op = optimizer.minimize(loss_fn)
sess = tf.Session()
# 参数初始化
sess.run(tf.global_variables_initializer())

batch_size = 128
n_samples = int(mnist.train.num_examples)
epochs = 10 
for epoch in range(epochs):
  avg_loss = 0
  total_batch = int(n_samples / batch_size)
  for i in range(total_batch):
    start_index = np.random.randint(0, len(X_train) - batch_size)
    batch_xs = X_train[start_index:(start_index + batch_size)]
    loss, _ = sess.run((loss_fn, optimize_op), feed_dict = {x:batch_xs} )
    avg_loss += loss / n_samples * batch_size
  print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss))

t_loss = sess.run(loss_fn, feed_dict = {x: X_test}) 
print("Test Data loss %f " % t_loss)

