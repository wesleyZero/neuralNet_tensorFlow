import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.keras import layers 
import numpy as np

(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

# Normalization of the dataset. 
# The data is 8-bit, normalized to a floating point value for the NN 
# The y-data is not normalized because it's just the integer labels
x_train = x_train / 255
x_val = x_val / 255

# Let's look at an image from the dataset 
print(type(x_train))
print(x_train.shape)
print(y_train.shape)
print(y_train[0])
# plt.imshow(x_train[0], cmap='hot')
# plt.show()

layer_input = layers.Input(x_train.shape[1:])
layer_hidden_0 = layers.Flatten()
layer_output = layers.Dense(10, activation='softmax')

nn_layers = [layer_input, layer_hidden_0, layer_output]
nn_model = tf.keras.models.Sequential(nn_layers)

nn_model.compile(optimizer='adam', loss='categorical_crossentropy')
print(nn_model.summary())

one_hot_y_train = tf.one_hot(y_train, 10)
nn_model.fit(x_train, one_hot_y_train)


print("The program ran")