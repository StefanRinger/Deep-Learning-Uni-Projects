from Optimization import Optimizers
from Layers import FullyConnected
from Optimization import Loss
import numpy as np
import unittest
import argparse


""" optimizer = Optimizers.Sgd(1.)
result = optimizer.calculate_update(1., 1.)
np.testing.assert_almost_equal(result, np.array([0.]))
result = optimizer.calculate_update(result, 1.)
np.testing.assert_almost_equal(result, np.array([-1.])) """

# input_size = 5
# batch_size = 10
# half_batch_size = int(batch_size / 2)
# input_tensor = np.ones([batch_size, input_size])
# input_tensor[0:half_batch_size, :] -= 2

# label_tensor = np.zeros([batch_size, input_size])
# for i in range(batch_size):
#     label_tensor[i, np.random.randint(0, input_size)] = 1

#Forward
# expected_tensor = np.zeros([batch_size, input_size])
# expected_tensor[half_batch_size:batch_size, :] = 1

#layer = ReLU.ReLU()
# output_tensor = layer.forward(input_tensor)
# print(np.sum(np.power(output_tensor - expected_tensor, 2)), 0)

#Backward

# expected_tensor = np.zeros([batch_size, input_size])
# expected_tensor[half_batch_size:batch_size, :] = 2

# layer = ReLU.ReLU()
# layer.forward(input_tensor)
# output_tensor = layer.backward(input_tensor * 2)
# print(np.sum(np.power(output_tensor - expected_tensor, 2)), 0)


# batch_size = 9
# categories = 4
# label_tensor = np.zeros([batch_size, categories])
# for i in range(batch_size):
#      label_tensor[i, np.random.randint(0, categories)] = 1

# input_tensor = np.zeros([batch_size, categories]) + 10000.
# layer = SoftMax.SoftMax()
# pred = layer.forward(input_tensor)
# input_tensor = label_tensor * 100.
# pred = layer.forward(input_tensor)
# error = layer.backward(label_tensor)

# batch_size = 9
# categories = 4
# label_tensor = np.zeros([batch_size, categories])
# for i in range(batch_size):
#      label_tensor[i, np.random.randint(0, categories)] = 1

# layer = Loss.CrossEntropyLoss()
# loss = layer.forward(label_tensor, label_tensor)
# print(label_tensor)


batch_size = 9
input_size = 4
output_size = 3
input_tensor = np.random.rand(batch_size, input_size)

categories = 4
label_tensor = np.zeros([batch_size, categories])
for i in range(batch_size):
    label_tensor[i, np.random.randint(0, categories)] = 1  # one-hot encoded labels



layer = FullyConnected.FullyConnected(input_size, output_size)
output_tensor = layer.forward(input_tensor)
# print(output_tensor.shape)
error_tensor = layer.backward(output_tensor)
print(error_tensor.shape[0])
# .assertEqual(error_tensor.shape[1], .input_size)
# .assertEqual(error_tensor.shape[0], .batch_size)
