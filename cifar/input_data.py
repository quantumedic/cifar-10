import tensorflow as tf

def read_decode(filename):
	filename_queue = tf.train.string_input_producer([filename])

	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example, features={
		'label': tf.FixedLenFeature([], tf.int64),
		'img_raw': tf.FixedLenFeature([], tf.string)
	})
	img_raw = tf.decode_raw(features['img_raw'], tf.uint8)
	img = tf.reshape(img_raw, [32, 32, 3])
	img = tf.cast(img, tf.float32)
	label = tf.cast(features['label'], tf.int32)
	label = tf.one_hot(label, 10, dtype=tf.int32)
	return img, label

def next_batch(size):
	image, label = read_decode('./records/train_data_batch_2.tfrecords')
	image_batch, label_batch = tf.train.shuffle_batch(
		[image, label],
		batch_size=size,
		capacity=10000,
		min_after_dequeue=size
	)
	return image_batch, label_batch

def test_batch():
	image, label = read_decode('./records/train_test_batch.tfrecords')
	image_batch, label_batch = tf.train.shuffle_batch(
		[image, label],
		batch_size=100,
		capacity=10000,
		min_after_dequeue=100
	)
	return image_batch, label_batch