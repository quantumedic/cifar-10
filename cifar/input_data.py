import tensorflow as tf

reader = tf.TFRecordReader()
train_queue = tf.train.string_input_producer(['./records/train_data_batch.tfrecords'])
test_queue = tf.train.string_input_producer(['./records/train_test_batch.tfrecords'])

def read_decode(train_mode):
	_queue = train_mode and train_queue or test_queue
	_, serialized_example = reader.read(_queue)
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
	image, label = read_decode(True)
	image_batch, label_batch = tf.train.shuffle_batch(
		[image, label],
		batch_size=size,
		capacity=50000,
		min_after_dequeue=size
	)
	return image_batch, label_batch

def test_batch():
	image, label = read_decode(False)
	image_batch, label_batch = tf.train.shuffle_batch(
		[image, label],
		batch_size=10000,
		capacity=10000,
		min_after_dequeue=1000
	)
	return image_batch, label_batch