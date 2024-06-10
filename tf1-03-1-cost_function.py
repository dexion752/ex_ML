# 비용함수 만들기
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
# our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())
# Variable for plotti8ng coat function

W_val = []
cost_val = []
for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# show the cost function
plt.plot(W_val, cost_val)
plt.show()



