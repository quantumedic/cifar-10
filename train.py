import argparse
import sys
import tempfile
import tensorflow as tf
from model.optimize import optimize
from cifar import feed
import deepnn
import saver_dev
import saver_prod

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
	model_saver = tf.saved_model.builder.SavedModelBuilder('./dist')
	model_saver.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])
	
	with tf.name_scope('train'):
		# try restore saver
		saver_dev.restore(train_saver, sess)

		x_train, y_train, k_train = feed.feed_data(True, 0.5, sess)
		x_test, y_test, k_test = feed.feed_data(False, 1, sess)

		for i in range(1000):
			if i % 10 == 0:
				x_batch, y_batch = sess.run([x_test, y_test])
				summary, acc = sess.run([merged, accuracy], feed_dict={x: x_batch, y_: y_batch, keep_prob: k_test})
				test_writer.add_summary(summary, i)
				saver_dev.save(train_saver, sess)
				print('Accuracy at step %s: %s' % (i, acc))
			else:
				x_batch, y_batch = sess.run([x_train, y_train])
				summary, _ = sess.run([merged, train_step], feed_dict={x: x_batch, y_: y_batch, keep_prob: k_train})
				train_writer.add_summary(summary, i)

		x_batch, y_batch = sess.run([x_test, y_test])
		saver_prod.save(model_saver, sess)
		print('test accuracy %g' % accuracy.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: k_test}))

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