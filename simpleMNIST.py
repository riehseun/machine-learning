import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_date

# we use TF helper function to pull down data from MNIST site
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x is placeholder for 28 x 28 image data
x = tf.placeholder(tf.float32, shape=[None, 784])

# y_ is called "y bar" and is a 10 element vector, containing predicted probability of each
# digit(0-9) class
y_ = tf.placeholder(tf.float32, [None, 10])

# define weights and balances
W = tf.Variables(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define our model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss is cross entropy
cross_entropy = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# each training step in gradient descent we want to minimize cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialize the global variable
init = tf.global_variables_initializer()

# create an interactive session that can span multiple code blocks
# Don't forget to explicitly close the session with sess.close()
sess = tf.Session()

# perform initialization which is only initialization of all global variables
sess.run(init)

# perfrom 1000 training steps
for i in range(1000):
	batch_xs batch_ys = mnist.train.next_batch(100) # get 100 random data points from data. batch xs = image, batch ys = digit(0-9) class
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # do optimization with this data

# evaluate how well the model did. Do this by comparing digit with the highest probability in actual (y) and predicted (y_)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf_float32))
test_accuracy = sess.run(accuracy, feed_dict={x: minst.test.images, y_: minst.test.labels})
print("Test Accuray: {0}%".format(test_accuracy * 100.0))

sess.close()