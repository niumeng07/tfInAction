import tensorflow as tf 
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
mnist = input_data.read_data_sets("/Users/liuda/Local/data/mnist/", one_hot = True)

def xavier_init(fan_in, fan_out, constant = 1):
  low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
  high = constant * np.sqrt(6.0 / (fan_in + fan_out))
  return tf.random_uniform((fan_in, fan_out),
                         minval = low, maxval = high,
                         dtype = tf.float32)


# feature structor
input_units= 784
hide1_units= 300
category = 10
W1 = tf.Variable( xavier_init(input_units, hide1_units))
b1 = tf.Variable( xavier_init(1, hide1_units))
W2 = tf.Variable( xavier_init(hide1_units, category))
b2 = tf.Variable( xavier_init(1, category))

x = tf.placeholder(tf.float32, [None, input_units])
y_ = tf.placeholder(tf.float32, [None, category])
drop_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
hidden1_drop = tf.nn.dropout( hidden1, drop_prob)

y = tf.nn.softmax( tf.add(tf.matmul(hidden1_drop, W2), b2))

loss_fn = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
#train = tf.train.AdagradOptimizer(0.3).minimize(loss_fn)
train = tf.train.AdamOptimizer(0.001).minimize(loss_fn)

global_init_op = tf.global_variables_initializer()
#sess = tf.Session()
sess = tf.InteractiveSession()
sess.run(global_init_op)

for i in range(5000):
    xs, ys = mnist.train.next_batch(100)
    train.run({x:xs, y_:ys, drop_prob:0.75})
    if i % 300 == 0:
        correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        print(accuracy.eval({ x: mnist.test.images, y_:mnist.test.labels, drop_prob: 1.0}))

print("accuracy")
correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
print(accuracy.eval({ x: mnist.test.images, y_:mnist.test.labels, drop_prob: 1.0}))
