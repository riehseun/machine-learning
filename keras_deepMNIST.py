import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K

# create input object which reads data from MNIST datasets. Perform one-hot encoding to define the digit
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# using interative session makes it the default sessions so we do not need to pass sess
sess = tf.InteractiveSession()

image_rows = 28
image_cols = 28

# reshape training and test images to 28 x 28 x 1
train_images = mnist.train.images.reshape(mnist.train.images.shape[0], image_rows, image_cols, 1)
test_images = mnist.test.images.reshape(mnist.test.images.shape[0], image_rows, image_cols, 1)

# layer values
num_filters = 32 # number of conv filters
max_pool_size = (2, 2) # shape of MaxPool
conv_kernel_size = (3, 3) # conv kernel shape
imag_shape = (28, 28, 1)
num_classes = 10
drop_prob = 0.5 # fraction to drop (0-1.0)

# define model type
model = Sequential()
