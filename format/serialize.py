import tensorflow as tf
from PIL import Image
import unpack

def init_feature(type, value):
	if type == 'int64':
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
	else:
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def build(path, filename, mode):
	if mode == 'train':
		writer = tf.python_io.TFRecordWriter('../records/train_data_batch.tfrecords')
	else:
		writer = tf.python_io.TFRecordWriter('../records/train_' + filename + '.tfrecords')

	data, labels = unpack.extract(path + filename)
	for i in range(len(data)):
		print('extract file index of %u from %s' % (i, filename))
		img = Image.fromarray(unpack.read(data[i]))
		img_raw = img.tobytes()
		example = tf.train.Example(features=tf.train.Features(feature={
			'label': init_feature('int64', labels[i]),
			'img_raw': init_feature('bytes', img_raw)
		}))
		writer.write(example.SerializeToString())
	print('record %s complete' % (filename))
	writer.close()