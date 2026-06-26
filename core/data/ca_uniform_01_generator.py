from __future__ import absolute_import, division, print_function

import numpy as np

from core.base.base_generator_ca import BaseGenerator

class Generator(BaseGenerator):
    def __init__(self, config, mode, X = None, ADV = None):
        super(Generator, self).__init__(config, mode)
        self.build_generator(X = X, ADV = ADV)

    def generate_random_X(self, shape):
        return np.random.uniform(0.0, 1.0, size = shape)

    def generate_random_ADV(self, shape):
        return np.random.uniform(0.0, 1.0, size = shape)
    
    def generate_random_C(self, shape):
        # return np.random.uniform(-1.0, 1.0, size = shape)
        return np.zeros(shape)