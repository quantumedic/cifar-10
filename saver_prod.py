import os
import tensorflow as tf

def restore(saver, sess):
	if os.path.exists('./save/checkpoint'):
		saver.restore(sess, './save/model.ckpt')

def save(saver, sess):
	saver.save()