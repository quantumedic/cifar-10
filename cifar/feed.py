import tensorflow as tf
from . import input_data

coord = tf.train.Coordinator()

def feed_data(train_mode, keep_prob, sess):
	if train_mode:
		xs, ys = input_data.next_batch(50)
		k = keep_prob
	else:
		xs, ys = input_data.test_batch()
		k = 1.0

	with tf.name_scope('queue'):
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	with tf.name_scope('read'):
		x_batch, y_batch = sess.run([xs, ys])

	return x_batch, y_batch, k