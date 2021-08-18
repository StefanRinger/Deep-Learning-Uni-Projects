import numpy as np

class BaseLayer:
    def __init__(self, isTrainable = False, testing_phase=False):
        self.trainable = isTrainable
        self.testing_phase = testing_phase