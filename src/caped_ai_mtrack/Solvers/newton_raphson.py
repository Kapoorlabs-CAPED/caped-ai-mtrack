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

    def updatePowCache(self, xc: float):

        for j in range(-3, self.degree):
            self.powcache[j + 3] = math.pow(self.xc, j)

    # def run(self):

    def distance(self, minx, miny, maxx, maxy):

        distance = (maxx - minx) * (maxx - minx) + (maxy - miny) * (
            maxy - miny
        )

        return np.sqrt(distance)

    def computeFunctions(self):

        for j in range(0, self.degree):

            c = self.coeff[j]
            self.polyfunc = self.polyfunc + c * self.powcache[j + 3]

            c = c * (j + self.degree)
            self.polyfuncdiff = self.polyfuncdiff + c * self.powcache[j + 2]

            c = c * (j + self.degree - 1)
            self.delpolyfuncdiff = (
                self.delpolyfuncdiff + c * self.powcache[j + 1]
            )

    def iterate(self, function, functionderiv, functionsecondderiv):

        self.xcnew = self.xc - (function / functionderiv) * (
            1
            + 0.5
            * function
            * functionsecondderiv
            / (functionderiv * functionderiv)
        )
