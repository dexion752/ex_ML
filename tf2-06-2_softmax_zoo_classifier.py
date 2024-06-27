import tensorflow as tf
import numpy as np

# Predicting animal type based on variou features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

nb_classes = 7

# Convert y_data to one_hot
y_one_hot = tf.keras.utils.to_categorical(y_data, nb_classes)
print("one_hot", y_one_hot)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=16, activation='softmax'))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_data, y_one_hot, epochs=1000)

# Single data test
test_data = np.array([[0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0]]) # expected prediction == 3 (feathers)
print(tf.model.predict(test_data), np.argmax(tf.model.predict(test_data), axis=-1))

# Full x_data test
pred = np.argmax(tf.model.predict(x_data), axis=-1)
for p, y in zip(ped, y_data.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
