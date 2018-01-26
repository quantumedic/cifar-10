import tensorflow as tf

from model import layer
from model import variable

def build(x):
	# reshape
	with tf.name_scope('reshape'):
		x_reshaped = tf.reshape(x, [-1, 32, 32, 3])
		tf.summary.image('input_image', x_reshaped, 10)


	# first convolutional layer
	with tf.name_scope('conv1'):
		W_conv1 = variable.init_weight([5, 5, 3, 32])
		b_conv1 = variable.init_bias([32])
		h_conv1 = tf.nn.relu(layer.init_conv(x_reshaped, W_conv1) + b_conv1)
		# summary.record_scalar(W_conv1)


	# layer.init_pooling
	with tf.name_scope('pool1'):
		h_pool1 = layer.init_pooling(h_conv1)

	# second full convolutional layer
	with tf.name_scope('conv2'):
		W_conv2 = variable.init_weight([5, 5, 32, 64])
		b_conv2 = variable.init_bias([64])
		h_conv2 = tf.nn.relu(layer.init_conv(h_pool1, W_conv2) + b_conv2)
		# summary.record_scalar(W_conv2)

	# second layer.init_pooling
	with tf.name_scope('pool2'):
		h_pool2 = layer.init_pooling(h_conv2)

	# second full connected layer
	with tf.name_scope('fc1'):
		h_pool2_flated = tf.reshape(h_pool2, [-1, 8 * 8 * 64])

		W_fc1 = variable.init_weight([8 * 8 * 64, 1024])
		b_fc1 = variable.init_bias([1024])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flated, W_fc1) + b_fc1)

	# dropout
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_dropped = tf.nn.dropout(h_fc1, keep_prob)
		

	# softmax layer
	with tf.name_scope('softmax'):
		W_output = variable.init_weight([1024, 10])
		b_output = variable.init_bias([10])

		y = tf.matmul(h_fc1_dropped, W_output) + b_output
		# tf.summary.histogram('W_output', W_output)
		# tf.summary.histogram('b_output', b_output)
		# tf.summary.histogram('y_output', y)

	return y, keep_prob