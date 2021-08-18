from Optimization import Optimizers
from Layers import *
from Optimization import *
import numpy as np




batch_size = 2
input_shape = (2, 4, 7)
input_size = np.prod(input_shape)

np.random.seed(1337)
input_tensor = np.random.uniform(-1, 1, (batch_size, *input_shape))

categories = 12
label_tensor = np.zeros([batch_size, categories])
for i in range(batch_size):
    label_tensor[i, np.random.randint(0, categories)] = 1

layers = list()
layers.append(None)
layers.append(Flatten.Flatten())



label_tensor = np.random.random((batch_size, 24))
layers[0] = Pooling.Pooling((2, 1), (2, 2))
difference = Helpers.gradient_check(layers, input_tensor, label_tensor)
print(difference)
#assertLessEqual(np.sum(difference), 1e-6)


