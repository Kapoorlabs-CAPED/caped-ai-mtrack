import numpy as np


class NewtonRaphson:
    def __init__(self, degree: float):

        self.degree = degree
        self.rndx = np.random.random()
        self.xc = np.random.random()
        self.xcnew = np.random.random()
        self.powcache = np.zeros(degree + 4)
