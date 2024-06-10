import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [1,2,3]
y_data = [1,2,3]

# W = tf.Variable(tf.random_normal([1]), name='weight')
# W = tf.Variable(5.0)
W = tf.Variable(-3.0)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
# cost = tf.reduce_sum(tf.square(hypothesis - Y))
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimizing: Gradient Descent using derivative: W -= learning_rate * derivative
# 직접 구현한 경우
# learning_rate = 0.1
# gradient = tf.reduce_mean((W * X - Y) * X)
# descent = W - learning_rate * gradient
# update = W.assign(descent)

# 내장 함수(.train.BradientDescentOptimizer())를 이용한 코드
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Lanuch the graph in a session
sess = tf.Session()

# Initialize global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(100):
    # sess.run(update, feed_dict={X: x_data, Y: y_data})
    # print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
    print(step, sess.run(W))
    sess.run(train, feed_dict={X: x_data, Y: y_data})
