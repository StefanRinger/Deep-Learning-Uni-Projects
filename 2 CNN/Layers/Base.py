import numpy as np

class BaseLayer:
    def __init__(self, isTrainable = False):
        self.trainable = isTrainable
        self.weights = None