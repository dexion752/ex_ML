# tensorflow 2.x 에서 1.x 코드 실행을 위항 처리
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

# 1. 학습 데이터 입력
# X = [1,2,3]
# Y = [1,2,3]
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])
# 2. 가중치. 편향 변수 설정 및 랜덤 수치 입력
# 1차원 행렬을 랜듬하게 입력: .randon_normal([1])
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 3. 선형 변수 함수 설정
hypothesis = X * W + b

# 4. 비용/손실 함수 설정
# - 평균제곱오차(mse) 사용
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 5. 옵티마이저 설정
# - train.GradientDescentOptimizer() 사용
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# 6. 비용함수를 최소화하기 위한 학습 설정
train = optimizer.minimize(cost)

# 연산 과정
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    curr_cost, curr_W, curr_B, _ = \
        sess.run([cost, W, b, train], feed_dict={X: [1,2,3,4,5], Y: [2.1,3.1,4.1,5.1, 6.1]})

    if step % 20 == 0:
        print(step, curr_cost, curr_W, curr_B)


# testion our model
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))


