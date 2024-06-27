import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# print(x_data.shape)
# (759, 8)
# print(y_data.shape)
# (759,)

# placeholers for a tensor
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bais'))

# Hypothesis
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicated = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicated, Y), dtype=tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    feed = {X: x_data, Y: y_data}
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict=feed)
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, cost, accuracy], feed_dict=feed)
    print("\nHypothesis: ", h, "\ncorrect (Y): ", c, "\nAccuracy: ", a)


