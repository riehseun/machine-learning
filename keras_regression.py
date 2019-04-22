#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import math

from keras.models import Sequential
from keras.layers.core import Dense, Activation

# generate some house sizes b/t 1000 and 3500
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# generate house prices from house size witha random noise added
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

# you need to normalize values to prevernt under/overflows
def normalize(array):
	return (array - array.mean()) / array.std()

# define number of training samples, 0.7 = 70%. We can take the first 70% since the values are randomized
num_train_samples = math.floor(num_house * 0.7)

# define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# define test data
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

# define a gradient descent optimizer that will minimize the loss defined in the operation "cost"
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# define the NN for doing linear regression
model = Sequential()
model.add(Dense(1, input_shape(1,), init='uniform', activation='linear'))
model.compile(loss='mean_squared_error', optimizer='sgd') # loss and optimizer

# fit/train the model
model.fit(train_house_size_norm, train_price_norm, nb_epoch=300)

# note: fit cost values will be different because we did not use NN in original
score = model.evaluate(test_house_size_norm, test_house_price_norm)
print("\nloss on test: {0}".format(score))