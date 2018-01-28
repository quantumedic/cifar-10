import tensorflow as tf
from . import input_data

def feed_data(train_mode, keep_prob, sess):
	if train_mode:
		xs, ys = input_data.next_batch(50)
		k = keep_prob
	else:
		xs, ys = input_data.test_batch()
		k = 1.0

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	return xs, ys, k