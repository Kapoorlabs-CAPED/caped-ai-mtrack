import math

import numpy as np


class NewtonRaphson:
    def __init__(self, degree: float, coeff: np.ndarray):

        self.degree = degree
        self.coeff = coeff
        self.rndx = np.random.random()
        self.xc = np.random.random()
        self.xcnew = np.random.random() * np.random.random()
        self.powcache = np.zeros(degree)
        self.polyfunc = 0
        self.polyfuncdiff = 0
        self.delpolyfuncdiff = 0
        self.secdelpolyfuncdiff = 0
        self.MAX_ITER = 1000
        self.MIN_CHANGE = 1.0e-1

    def _updatePowCache(self, xc: float):

        for j in range(self.degree):

            self.powcache[j] = math.pow(xc, self.degree - j)

    def run(self, x, y):

        self._updatePowCache(self.xc)
        self._computeFunctions()
        iteration = 0

        while abs(self.xcnew - self.xc) > self.MIN_CHANGE:

            self.xc = self.xcnew

            iteration = iteration + 1

            self._iterate(self.polyfunc, self.polyfuncdiff)

            if math.isnan(self.xcnew):
                self.xcnew = self.xc

            self._updatePowCache(self.xcnew)
            self._computeFunctions()

            if iteration >= self.MAX_ITER:
                break

        self.polyfunc = 0
        for j in range(self.degree):
            self.polyfunc = self.polyfunc + self.coeff[j] * math.pow(
                self.xc, self.degree - j
            )
        return self._distance(x, y, self.xc, self.polyfunc)

    def _distance(self, minx, miny, maxx, maxy):

        distance = (maxx - minx) * (maxx - minx) + (maxy - miny) * (
            maxy - miny
        )

        return np.sqrt(distance)

    def _computeFunctions(self):

        self.polyfunc = 0
        for j in range(0, self.degree):

            c = self.coeff[j]
            self.polyfunc = self.polyfunc + c * self.powcache[j]

            c = c * (self.degree - j)
            self.polyfuncdiff = self.polyfuncdiff + c * self.powcache[j - 1]

    def _iterate(self, function, functionderiv):

        self.xcnew = self.xc - (function / functionderiv)
