# 1. Import Libraries for Data Engineering

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# 2. Load MNIST dataset

mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# MNIST Dataset을 TensorFlow(Keras) 에서 load하게 되면 28x28의 크기로 load하게 됩니다.


# 3. Split data
X_train, x_val, Y_train, t_val = train_test_split(X_train, Y_train, test_size=0.2)
# 기계 학습 작업을 수행할지 여부에 관계없이 데이터 세트를 분할하는 데 유용한 Scikit-learn의 train_test_split 모듈을 사용

# 4. Data Preprocessing
# Flattening, Normalization, One-Hot Encoding
X_train = (X_train.reshape(-1, 784) / 255).astype(np.float32)
# reshape()의 '-1'은 변경된 배열의 '-1' 위치의 차원은 "원래 배열의 길이와 남은 차원으로부터 추정"이 된다는 뜻
# 따라서 (-1, 784)는 열은 784개로 고정하고 행은 여기에 맞추어 추정(계산)하라는 의미이다.
# 물론 784는 이미지가 28X28 픽셀 사이즈이기 때문이다.
X_test = (X_test.reshape(-1, 784) / 255).astype(np.float32)
x_val = (x_val.reshape(-1, 784) / 255).astype(np.float32)
Y_train = np.eye(10)[Y_train].astype(np.float32)
Y_test = np.eye(10)[Y_test].astype(np.float32)
t_val = np.eye(10)[t_val].astype(np.float32)

# 5. Import Libraries for Model Engineering

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

np.random.seed(123)
tf.random.set_seed(123)

# 6. Set Hyperparameter
hidden_size = 200
output_dim = 10  # output layer dimensionality
EPOCHS = 30
batch_size = 100
learning_rate = 5e-4

# 7. Build NN model
class Feed_Forward_Net(Model):
    '''
    Multilayer perceptron
    '''
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.l1 = Dense(hidden_size, activation='sigmoid')
        self.l2 = Dense(hidden_size, activation='sigmoid')
        self.l3 = Dense(hidden_size, activation='sigmoid')
        self.l4 = Dense(output_dim, activation='softmax')

    def call(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        y = self.l4(h3)

        return y

model = Feed_Forward_Net(hidden_size, output_dim)

# 8. Optimizer
optimizer = optimizers.SGD(learning_rate=learning_rate)
# optimizer = optimizers.Adam(learning_rate=learning_rate)


# 9. Model compilation - model.compile
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 10. Train & Validation
model.fit(X_train, Y_train,
          epochs=EPOCHS, batch_size=batch_size,
          validation_data=(x_val, t_val))

# 11. Assess model performance
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('test_loss: {:.3f}, test_acc: {:.3f}'.format(loss, acc))


