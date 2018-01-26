import tensorflow as tf

def record_scalar(value):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(value)
		tf.summary.scalar('mean', mean)

		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(value - mean)))

		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(value))
		tf.summary.scalar('min', tf.reduce_min(value))
		tf.summary.histogram('histogram', value)