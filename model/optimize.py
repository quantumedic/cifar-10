import tensorflow as tf
from . import summary

def optimize(y_, y):
	with tf.name_scope('loss'):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

	cross_entropy = tf.reduce_mean(cross_entropy)
	summary.record_scalar(cross_entropy)

	with tf.name_scope('optimizer'):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	with tf.name_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		correct_prediction = tf.cast(correct_prediction, tf.float32)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	summary.record_scalar(accuracy)
	return accuracy, train_step