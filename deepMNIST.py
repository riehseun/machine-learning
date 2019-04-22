import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# create input object which reads data from MNIST datasets. Perform one-hot encoding to define the digit
mnist= input_data.read_data_sets("MNIST_data/", one_hot=True)

# using interative session makes it the default sessions so we do not need to pass sess
sess = tf.InteractiveSession()

# define placeholders for MNIST input data
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# change the MNIST input data from a list of values to a 28 pixel X 1 grayscale value cube
# which the Convolution NN can use
x_image = tf.reshape(x, [-1,28,28,1], name="x_image")

# define helper functions to create weights and biases variables, and convolutions, and pooling layers
# we are using RELU as our activation function. These must be initialized to a small positive number
# and with some noise so you don't end up going to zero when comparing diffs
def weight_variable(shape, name=None):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

# convolution and pooling - we do convolution, and then pooling to control overfitting
def conv2d(x, W, name=None):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)

def max_pool_2x2(x, name=None):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)

# define layers in NN

# 1st convolution layer
with tf.name_scope('Conv1')
	# 32 features for each 5x5 patch of the image
	W_conv1 = weight_variable([5,5,1,32], name="weight")
	b_conv1 = bias_variable([32] name="bias")
	# do convolution on images, add bias and push through RELU activation
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name="relu")
	# take results and run through max_pool
	h_pool1 = max_pool_2x2(h_conv1, name="pool")

# 2nd convolution layer
with tf.name_scope('Conv2')
	# process the 32 features from convolution layer 1, in 5 x 5 patch. Return 64 features weights and biases
	W_conv2 = weight_variable([5,5,32,64])
	b_conv2 = bias_variable([64])
	# do convolution of output of 1st convolution layer. Rool results
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('FC')
	# fully connected layer
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	# connect output of pooling layer 2 as input to full connected layer
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout some neurons to reduce overfitting
keep_prob = tf.placeholder(tf.float32) # get dropout probability as a training input
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('Readout')
	# readout layer
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

# define model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# loss measurement
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# loss optimization
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# what is correct
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

# how accurate is it?
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize all of the variables
sess.run(tf.global_variables_initializer())

# train the model
import time

# define number of steps and how often we display progress
num_steps = 3000
display_every = 100

# start timer
start_time = time.time()
end_time = time.time()

for i in range(num_steps):
	batch = mnist.train.next_batch(50)
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	# periodic status display
	if i%display_every == 0:
		train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		end_time = time.time()
		print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100))

# display summary
# time to train
end_time = time.time()
print("Total training time for {0} batches: {1:.2f} seconds".format(i+1, end_time-start_time))

# accuracy on test data
print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100.0))

sess.close()