#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import math
import tflearn

# generate some house sizes b/t 1000 and 3500
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# generate house prices from house size witha random noise added
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

# plot house vs size
#plt.plot(house_size, house_price, "bx") # bx = blue x
#plt.ylabel("Price")
#plt.xlabel("Size")
#plt.show()

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

# one value, one value out.
input = tflearn.input_date(shape[None], name="InputData") # input data is a list of undefined length
linear = tflearn.layers.core.single_unit(input, activation='linear', name="Linear") # define a single neuran with linear activation

# define the optimizer, metric we try to optimize, and how we calculate loss
reg = tflearn.regression(linear, optimizer='sgd', loss='mean_square', metric='R2', learning_rate=0.01, name='regression') # set the learning rate, default is off

# define the model
model = tflearn.DNN(reg)

# traing the model with training data
model.fit(train_house_size_norm, train_price_norm, n_epoch=1000)

print("Training complete")
# output W and b for the trained linear equation
print(" Weights: W={0}, b={1}\n".format(model.get_weights(linear.W), model.get_weights(linear.b)))

# evaluate accuracy
print(" Accuracy {0} ".format(model.evaluate(test_house_size_norm, test_house_price_norm)))