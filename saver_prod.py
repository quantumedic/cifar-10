import os
import tensorflow as tf

def restore(saver, sess):
	if os.path.exists('./save/checkpoint'):
		saver.restore(sess, './save/model.ckpt')

def save(saver, sess):
	saver.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])
	saver.save()