import os

def restore(saver, sess):
	if os.path.exists('./save/checkpoint'):
		saver.restore(sess, './save/model.ckpt')

def save(saver, sess):
	path = saver.save(sess, './save/model.ckpt')