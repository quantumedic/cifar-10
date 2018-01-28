import tensorflow as tf

def init_weight(shape):
	_init = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(_init)

def init_bias(shape):
	_init = tf.constant(0.1, shape=shape)
	return tf.Variable(_init)