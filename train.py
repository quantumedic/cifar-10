import argparse
import sys
import tempfile
import tensorflow as tf
from model.optimize import optimize
from cifar import feed
import deepnn
import saver

def train():
	sess = tf.InteractiveSession()

	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

	y, keep_prob = deepnn.build(x)

	accuracy, train_step = optimize(y_, y)

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
	test_writer = tf.summary.FileWriter('./logs/test')
	tf.global_variables_initializer().run()

	train_saver = tf.train.Saver()

	with tf.name_scope('train'):
		# try restore saver
		saver.restore(train_saver, sess)

		for i in range(100):
			if i % 10 == 0:
				xs, ys, k = feed.feed_data(False, 1, sess)
				summary, acc = sess.run([merged, accuracy], feed_dict={x: xs, y_: ys, keep_prob: k})
				test_writer.add_summary(summary, i)
				saver.save(train_saver, sess)
				print('Accuracy at step %s: %s' % (i, acc))
			else:
				xs, ys, k = feed.feed_data(False, 0.5, sess)
				summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_: ys, keep_prob: k})
				train_writer.add_summary(summary, i)

	xs, ys, k = feed.feed_data(False, 1, sess)
	print('test accuracy %g' % accuracy.eval(feed_dict={x: xs, y_: ys, keep_prob: k}))

	train_writer.close()
	test_writer.close()

def main(_):
	train()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--log_dir',
		type=str,
		default='./logs',
		help='Summaries log directory')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)