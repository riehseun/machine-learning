import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# create input object which reads data from MNIST datasets. Perform one-hot encoding to define the digit
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

image_rows = 28
image_cols = 28

# reshape the training and test images to 28 x 28 x 1
train_images = mnist.train.images.reshape(mnist.train.images.shape[0], image_rows, image_cols, 1)
test_images = mnist.test.images.reshape(mnist.test.images.shape[0], image_rows, image_cols, 1)

num_classes = 10
keep_prob = 0.5 # fraction to keep (0-1.0)

# define the shape of the data coming into the NN
input = input_data(shape=[None, 28, 28, 1], name='input')

# do convolution on images, add bias and push through RELU activation
network = conv_2d(input, nb_filter=32, filter_size=3, activation='relu', regularizer='L2')

# notice name was not specified. Mane is defaulted to "Conv2D" and will be postfixed with "_n"
# where n is the number of the occurance.
# take results and run through max_pool
network = max_pool_2d(network, 2)

# 2nd convolution layer
# do convolution on images, add bias and push through RELU activation
network = conv_2d(network, nb_filter=64, filter_size=3, activation='relu', regularizer='L2')
# take results and run through max_pool
network = max_pool_2d(network, 2)

# fully connected layer
network = fully_connected(network, 128, activation='tanh')

# dropout some neurons to reduce overfitting
network = dropout(network, keep_prob)

# readout layer
network = fully_connected(network, 10, activation='softmax')

# set loss, measurement, optimizer
network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')

# training
num_epoch = 2 # number of times through the data
model = tflearn.DNN(network, tensorboard_verbose=0) # for more info in tensorboard turn on tensorboard_verbose
model.fit({'input': train_images}, {'target': mnist.train.labels}, n_epoch=num_epoch)
