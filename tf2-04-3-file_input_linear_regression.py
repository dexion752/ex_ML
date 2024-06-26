import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data, "\nx_data shape: ",x_data.shape)
print(y_data, "\ny_data shape: ",y_data.shape)

tf.model = tf.keras.Sequential()
# activation function doesn't have to be added as s separate layer.
# Add it as an argument of Dense() layer
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3, activation='linear'))
# tf.model.add(tf.keras.layers.Activation('linear'))
tf.model.summary()


tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5))
history = tf.model.fit(x_data, y_data, epochs=2000)

# Ask my score
print("Your score will be ", tf.model.predict(np.array([[100, 70, 101]])))
print("Other scores will be ", tf.model.predict(np.array([[60, 70, 110], [90, 100, 80]])))
