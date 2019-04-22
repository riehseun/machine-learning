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

# define layers in NN
# define the 1st convolution layer. we use border_mode = and input_shape_only on first layer
# border_mode =value restricts convolution to only where the input and the filter fully overlap (i.e. not partial overlap)
model.add(Convolution2D(num_filters, conv_kernel_size[0], conv_kernel_size[1], border_mode='valid', input_shape=img_shape))

# push through RELU activation
model.add(Activation('relu'))
# take results and run through max_pool
model.add(MaxPooling2D(pool_size=max_pool_size))

# 2nd convolutionn layer
model.add(Convolution2D(num_filters, conv_kernel_size[0], conv_kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=max_pool_size))

# fully connected layer
model.add(Flatten())
model.add(Dense(128)) # fully connected layer in Keras
model.add(Activation('relu'))

# dropout some neurons to reducce overfitting
model.add(Dropout(drop_prob))

# readout layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# set loss and measurement, optimizer, and metric used to evaluate loss
model.compile(loss='categorical_crossentropy', optimizer='adm', metrics=['accuracy'])

# training settings
batch_size = 2
num_epoch = 2

# fit the training data to the model. Nicely displays the time, loss, and validation accuracy on the test data
model.fit(train_images, mnist.train.labels, batch_size=batch_size, nb_epoch=num_epoch,
		verbose=1, validation_data=(test_images, mnist.test.labels))