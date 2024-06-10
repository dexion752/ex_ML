import tensorflow as tf
import numpy as np

X = np.array([1,2,3])
Y = np.array([1,2,3])

# 1. 사용할 모델을 결정한다.
# - keras가 tensorflow 내로 편입되었으므로
# - . 접근자로 keras를 호출한 다음 선형 모델 Sequential() 호출
tf.model = tf.keras.Sequential()

# 2. '.add()'로 지정한 모델에 레이어 쌓기
# - 여기에서는 단층 레이어 구조입.
# - 출력할 행렬 차원은 units= 로 지정, 입력할 행력 차원은 imput_dim= 파라미터로 지정
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

# 3. 옵티마이저 지정
# - sgd 변수에 옵티마이저로 표준하강경사(SGD, standard gradient descent)를 지정하고
# - 학습률(learning_rate=)를 0.1로 설정
sgd = tf.keras.optimizers.SGD(learning_rate=0.1)

# 4. 모델 컴파일
# - 설정한 내용으로 모델 구조 완성
# - 이때 비용함수와 앞에서 설정한 옵티마이저 지정
# - 여기에서는 비용함수로 평균제곱오차(mse, mean square error) 사용
tf.model.compile(loss='mse', optimizer=sgd)

# 5. 모델 개괄
tf.model.summary()

# 6. 실제 데이더를 먹여서 학섭 진행
tf.model.fit(X, Y, epochs=200)

# 7. 결괏값을 이용하여 예측 시도
y_predict = tf.model.predict(np.array([5,4]))
print(y_predict)
# [[4.9962497] [3.9976254]]
