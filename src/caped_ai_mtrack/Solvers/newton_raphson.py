import math

import numpy as np


class NewtonRaphson:
    def __init__(self, degree: float, coeff: np.ndarray):

        self.degree = degree
        self.coeff = coeff
        self.rndx = np.random.random()
        self.xc = np.random.random()
        self.xcnew = np.random.random()
        self.powcache = np.zeros(degree + 4)
        self.polyfunc = 0
        self.polyfuncdiff = 0
        self.delpolyfuncdiff = 0
        self.secdelpolyfuncdiff = 0
        self.MAX_ITER = 1000000
        self.MIN_CHANGE = 1.0e-3

    def _updatePowCache(self, xc: float):

        for j in range(-3, self.degree):
            self.powcache[j + 3] = math.pow(xc, j)

    def run(self, x, y):

        self._updatePowCache(self.xc)
        self._computeFunctions()
        iteration = 0

        while abs(self.xcnew - self.xc) > self.MIN_CHANGE:

            self.xc = self.xcnew
            dmin = (self.polyfunc - y) * self.polyfuncdiff + (self.xc - x)
            dmindiff = (
                self.polyfuncdiff * self.polyfuncdiff
                + (self.polyfunc - y) * self.delpolyfuncdiff
                + 1
            )
            dminsecdiff = (
                (self.polyfunc - y) * self.secdelpolyfuncdiff
                + self.delpolyfuncdiff * self.polyfuncdiff
                + 2 * self.polyfuncdiff * self.delpolyfuncdiff
            )

            iteration = iteration + 1
            self._iterate(self.xc, dmin, dmindiff, dminsecdiff)
            if math.isnan(self.xcnew):
                self.xcnew = self.xc
            self.delpolyfuncdiff = 0
            self.polyfunc = 0
            self.polyfuncdiff = 0
            self.secdelpolyfuncdiff = 0

            self._updatePowCache(self.xcnew)
            self._computeFunctions()

            if iteration >= self.MAX_ITER:
                break

        self.polyfunc = 0
        for j in range(self.degree):
            self.polyfunc = self.polyfunc + self.coeff[j] * math.pow(
                self.xc, j
            )

        return self._distance(x, y, self.xc, self.polyfunc)

    def _distance(self, minx, miny, maxx, maxy):

        distance = (maxx - minx) * (maxx - minx) + (maxy - miny) * (
            maxy - miny
        )

        return np.sqrt(distance)

    def _computeFunctions(self):

        for j in range(0, self.degree):

            c = self.coeff[j]
            self.polyfunc = self.polyfunc + c * self.powcache[j + 3]

            c = c * (j + self.degree)
            self.polyfuncdiff = self.polyfuncdiff + c * self.powcache[j + 2]

            c = c * (j + self.degree - 1)
            self.delpolyfuncdiff = (
                self.delpolyfuncdiff + c * self.powcache[j + 1]
            )

            c = c * (j + self.degree - 2)
            self.secdelpolyfuncdiff = (
                self.secdelpolyfuncdiff + c * self.powcache[j]
            )

    def _iterate(self, function, functionderiv, functionsecondderiv):

        self.xcnew = self.xc - (function / functionderiv) * (
            1
            + 0.5
            * function
            * functionsecondderiv
            / (functionderiv * functionderiv)
        )
