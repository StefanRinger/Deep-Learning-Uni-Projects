import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.transform import rotate

import generator

label_path = 'Labels.json'
file_path = "exercise_data"



gen = generator.ImageGenerator(file_path, label_path, 60, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
b1 = gen.next()[0]
b2 = gen.next()[0]
np.testing.assert_almost_equal(b1[:20], b2[40:])