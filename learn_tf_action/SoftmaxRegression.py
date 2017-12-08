#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("./MNIST_data", one_hot = True)
mnist = input_data.read_data_sets("./Users/liuda/Local/data/mnist", one_hot = True)

def xavier_init(row, col, constant = 1): # 生成 row 行 col 列的随机均匀分布的矩阵
  low = -constant * np.sqrt(6.0 / (row + col))
  high = constant * np.sqrt(6.0 / (row + col))
  return tf.random_uniform((row, col),
                         minval = low, maxval = high,
                         dtype = tf.float32)


with tf.Session() as sess:
  sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses")
  x = tf.placeholder(tf.float32, [None, 784])

  W = tf.Variable(xavier_init(784, 10)) # 784行10列
  b = tf.Variable(xavier_init(1, 10))   # 1 行 10列

  y = tf.nn.softmax(tf.matmul(x, W) + b)

  y_ = tf.placeholder(tf.float32, [None, 10])
  loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
  train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # tf.global_variables_initializer().run()
  variable_init = tf.global_variables_initializer()
  sess.run(variable_init)

  for i in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      train.run({x: batch_xs, y_: batch_ys})

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
