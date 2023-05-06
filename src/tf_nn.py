import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.keras import layers 
import numpy as np

(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

# Normalization of the dataset. 
# The data is 8-bit, normalized to a floating point value for the NN

x_train, y_train, x_val, y_val = x_train / 255, y_train / 255, \
                                    x_val / 255, y_val / 255

print("The program ran")