import tensorflow as tf
from model import summary
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

		summary.record_scalar(W_conv1)

	# layer.init_pooling
	with tf.name_scope('pool1'):
		h_pool1 = layer.init_pooling(h_conv1)

	# second full convolutional layer
	with tf.name_scope('conv2'):
		h_dconv1 = layer.add_dconv(h_pool1, 32)

		W_conv2 = variable.init_weight([3, 3, 32, 64])
		b_conv2 = variable.init_bias([64])
		h_conv2 = tf.nn.relu(layer.init_conv(h_dconv1, W_conv2) + b_conv2)

		h_dconv2 = layer.add_dconv(h_conv2, 64)
		summary.record_scalar(W_conv2)

	with tf.name_scope('conv3'):
		h_dconv3 = layer.add_dconv(h_dconv2, 64)

		W_conv3 = variable.init_weight([3, 3, 64, 128])
		b_conv3 = variable.init_bias([128])
		h_conv3 = tf.nn.relu(layer.init_conv(h_dconv3, W_conv3) + b_conv3)

		h_dconv4 = layer.add_dconv(h_conv3, 128)
		summary.record_scalar(W_conv2)

	# second layer.init_pooling
	with tf.name_scope('pool2'):
		h_pool2 = layer.init_pooling(h_dconv4)

	# second full connected layer
	with tf.name_scope('fc1'):
		h_dconv4_flated = tf.reshape(h_pool2, [-1, 8 * 8 * 128])

		W_fc1 = variable.init_weight([8 * 8 * 128, 1024])
		b_fc1 = variable.init_bias([1024])
		h_fc1 = tf.nn.relu(tf.matmul(h_dconv4_flated, W_fc1) + b_fc1)

	
	# dropout
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_dropped = tf.nn.dropout(h_fc1, keep_prob)
		

	# softmax layer
	with tf.name_scope('softmax'):
		W_output = variable.init_weight([1024, 10])
		b_output = variable.init_bias([10])

		y = tf.matmul(h_fc1_dropped, W_output) + b_output
		tf.summary.histogram('W_output', W_output)
		tf.summary.histogram('b_output', b_output)
		tf.summary.histogram('y_output', y)

	return y, keep_prob
