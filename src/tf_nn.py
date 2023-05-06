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
plt.imshow(x_train[0], cmap='hot')
plt.show()



print("The program ran")