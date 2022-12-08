import math

import numpy as np


class NewtonRaphson:
    def __init__(
        self, degree: float, coeff: np.ndarray, x_1: float, y_1: float
    ):

        self.degree = degree
        self.coeff = coeff
        self.x_1 = x_1
        self.y_1 = y_1
        self.xc = abs(np.random.random())
        self.maxiter = 100
        self.delta = 0.01

    def func(self, x):

        return (
            (self.polyfunc - self.y_1) * self.polyfuncdiff + self.xc - self.x_1
        )

    def derivfunc(self, x):

        return (
            (self.polyfunc - self.y_1) * self.delpolyfuncdiff
            + self.polyfuncdiff * self.polyfuncdiff
            + 1
        )

    def run(self):

        self._computeFunctions()

        self.iter = 0
        while abs(self.func(self.xc)) > self.delta:
            self.iter = self.iter + 1
            self.xcnew = self.xc - (
                self.func(self.xc) / self.derivfunc(self.xc)
            )  # Newton-Raphson equation

            self.xc = self.xcnew
            self._computeFunctions()
            if self.iter > self.maxiter:
                break

        self.polyfunc = 0
        for j in range(self.degree):
            self.polyfunc = self.polyfunc + self.coeff[j] * math.pow(
                self.xc, self.degree - j
            )
        mindistance = self._distance(
            self.x_1, self.y_1, self.xc, self.polyfunc
        )

        return mindistance

    def _distance(self, minx, miny, maxx, maxy):

        distance = (maxx - minx) * (maxx - minx) + (maxy - miny) * (
            maxy - miny
        )

        return np.sqrt(distance)

    def _computeFunctions(self):

        self.polyfunc = 0
        self.polyfuncdiff = 0
        self.delpolyfuncdiff = 0
        for j in range(0, self.degree):

            if self.degree - j > 0:

                c = self.coeff[j]
                self.polyfunc = self.polyfunc + c * math.pow(
                    self.xc, self.degree - j
                )

                c = c * (self.degree - j)
                self.polyfuncdiff = self.polyfuncdiff + c * math.pow(
                    self.xc, self.degree - j - 1
                )

                c = c * (self.degree - j - 1)
                self.delpolyfuncdiff = self.delpolyfuncdiff + c * math.pow(
                    self.xc, self.degree - j - 2
                )
