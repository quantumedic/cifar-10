import serialize

for i in range(5):
	serialize.build('../dataset/', 'data_batch_' + str(i + 1))

serialize.build('../dataset/', 'test_batch')