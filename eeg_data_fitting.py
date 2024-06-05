# Importing the important libraries
import tensorflow as tf
import numpy as np

# Training Datasets
x_train = np.load("/home/rishabhj/Downloads/x_train.npy")
y_train = np.load("/home/rishabhj/Downloads/y_train.npy")

# x_train = x_train / np.max(x_train)


print(" The dimension of x_train is ", x_train.shape)
print(" The dimension of y_train is ", y_train.shape)

# Creation of model and compiling it.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(19, 500)),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
    loss = tf.keras.losses.MeanAbsoluteError(),
    metrics=['accuracy']

)

model.fit(x_train, y_train, epochs=20, verbose = 1)


# Loading the test data and checking the accuracy
x_test = np.load("/home/rishabhj/Downloads/x_test.npy")
y_test = np.load("/home/rishabhj/Downloads/y_test.npy")

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)
print("Test Loss : ", test_loss, "\n")
print("\n\n\n\nEverything OK HERE \n\n\n\n\n")