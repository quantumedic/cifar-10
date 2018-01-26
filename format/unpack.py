import pickle

def read(img):
	return img.transpose(1, 2, 0)

def extract(file):
	with open(file, 'rb') as dataset:
		_dict = pickle.load(dataset, encoding='bytes')
		labels = _dict[b'labels']
		data = _dict[b'data'].reshape(len(labels), 3, 32, 32)
	return data, labels