# optional: compute_gradient and apply_gradient
# 직접 계산한 그레디언트와 내장 그레디언트의 결과 비교
# step [매뉴얼, 웨이트, [(내장, 웨이트)]
# 0 [37.333332, 5.0, [(37.333336, 5.0)]]
# 1 [33.84889, 4.6266665, [(33.84889, 4.6266665)]]
# 2 [30.689657, 4.2881775, [(30.689657, 4.2881775)]]
# 3 [27.825287, 3.9812808, [(27.825287, 3.9812808)]]
# 4 [25.228262, 3.703028, [(25.228262, 3.703028)]]

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

X = [1,2,3]
Y = [1,2,3]

# set wrong model weight
W = tf.Variable(5.)
# Linear model
hypothesis = X * W
# Manual gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2
# Cost/lass function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# get gradients
gvs = optimizer.compute_gradients(cost)
# apply gradients: 이 단계에서 그레디언트를 직접 조작할 수 있다.
apply_gradients = optimizer.apply_gradients(gvs)

# Launch the graph in a session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)