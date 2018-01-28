import tensorflow as tf

def init_conv(input, filter):
	return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

def init_pooling(input):
	return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')