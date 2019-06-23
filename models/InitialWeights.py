# coding: utf-8

import numpy as np


class InitialWeights:
    def __init__(self):
        pass

    @classmethod
    def generate(cls, lines=1, columns=1, debug=False):
        if debug:
            return np.ones((lines, columns))
        else:
            return np.random.randn(lines, columns)


