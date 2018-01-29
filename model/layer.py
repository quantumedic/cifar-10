import tensorflow as tf
from . import variable

def init_conv(input, filter):
	return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

def init_pooling(input):
	return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def add_dconv(input, chanel):
	with tf.name_scope('dconv'):
		W_dconv = variable.init_weight([1, 1, chanel, chanel])
		b_dconv = variable.init_weight([chanel])

		h_dconv = tf.nn.relu(init_conv(input, W_dconv) + b_dconv)
	return h_dconv