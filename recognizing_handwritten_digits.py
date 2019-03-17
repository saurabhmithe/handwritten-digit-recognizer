import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path

# a neural network can be thought of as a function that maps the input to the output
# this function is also called as the 'model'
# we train this model over a set of data by showing it several examples
# if we know the expected correct output (labels), learning can be done in a supervised manner
# in case of no labels present, such learning is called as unsupervised learning


# build and return a model trained on the data set
def build_and_train_model(x, y):
    model = tf.keras.models.Sequential()  # sequential model
    model.add(tf.keras.layers.Flatten())  # input layer
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # hidden layer 1
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # hidden layer 2
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # output layer
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x, y, epochs=1)
    return model


# import mnist dataset - 28x28 images of handwritten digits from 0 - 9
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)

# load a pre-trained model if one exists, otherwise build and train a model and save it
model_file_name = 'model'
if os.path.isfile(model_file_name):
    trained_model = tf.keras.models.load_model(model_file_name)
else:
    trained_model = build_and_train_model(x_train, y_train)
    trained_model.save(model_file_name)

# make predictions using the trained model
x_test = tf.keras.utils.normalize(x_test, axis=1)
predictions = trained_model.predict(x_test)

# print the first predicted value
num = np.argmax((predictions[0]))
print(num)

# show the first hand-written digit
plt.imshow(x_test[0])
plt.show()

# evaluate the model metrics
val_loss, val_acc = trained_model.evaluate(x_test, y_test)
print(val_loss, val_acc)

