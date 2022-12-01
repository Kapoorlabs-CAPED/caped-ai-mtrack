import math

import numpy as np

from ..Solvers import NewtonRaphson
from .generalized_function import GeneralFunction


class QuadraticFunction(GeneralFunction):
    def __init__(self, points: np.ndarray, degree: int):

        super(GeneralFunction, self).__init__(points)
        self.num_points = self.get_num_points()
        self.degree = degree
        self.min_num_points = 3
        self.coeff = np.zeros(3)

    def fit(self):

        delta = np.zeros(9)
        theta = np.zeros(3)
        for i in range(self.num_points):

            point = self.points[i]

            y = point[0]
            x = point[1]

            xx = x * x
            xxx = xx * x

            delta[0] += xx * xx
            delta[1] += xxx
            delta[2] += xx

            delta[3] += xxx
            delta[4] += xx
            delta[5] += x

            delta[6] += xx
            delta[7] += x
            delta[8] += 1

            theta[0] += xx * y
            theta[1] += x * y
            theta[2] += y

        delta = np.invert(delta)

        self.coeff[0] = (
            delta[0] * theta[0] + delta[1] * theta[1] + delta[2] * theta[2]
        )
        self.coeff[1] = (
            delta[3] * theta[0] + delta[4] * theta[1] + delta[5] * theta[2]
        )
        self.coeff[2] = (
            delta[6] * theta[0] + delta[7] * theta[1] + delta[8] * theta[2]
        )

    def get_coefficients(self, j):

        return self.coeff[j]

    def predict(self, x):

        y = 0.0
        for j in range(self.degree):
            y = self.get_coefficients(j) + (x * y)
        return y

    def distance(self, point):

        x1 = point[1]
        y1 = point[0]

        a3 = 2 * self.coeff[0] * self.coeff[0]
        a2 = 3 * self.coeff[1] * self.coeff[0] / a3
        a1 = (
            2 * self.coeff[2] * self.coeff[0]
            - 2 * self.coeff[0] * y1
            + 1
            + self.coeff[1] * self.coeff[1]
        ) / a3
        a0 = (self.coeff[2] * self.coeff[1] - y1 * self.coeff[1] - x1) / a3

        p = (3 * a1 - a2 * a2) / 3
        q = (-9 * a1 * a2 + 27 * a0 + 2 * a2 * a2 * a2) / 27

        tmp1 = np.sqrt(-p / 3)
        tmp2 = q * q / 4 + p * p * p / 27

        if tmp2 > 0:

            aBar = np.cbrt(-q / 2 + np.sqrt(q * q / 4 + p * p * p / 27))
            bBar = np.cbrt(-q / 2 - np.sqrt(q * q / 4 + p * p * p / 27))

            xc1 = xc2 = xc3 = aBar + bBar - a2 / 3

        elif tmp2 == 0:

            if q > 0:

                xc1 = -2 * tmp1
                xc2 = tmp1
                xc3 = xc2

            elif q < 0:

                xc1 = 2 * tmp1
                xc2 = -tmp1
                xc3 = xc2

            else:

                xc1 = 0
                xc2 = 0
                xc3 = 0

        else:

            if q >= 0:
                phi = math.acos(-np.sqrt(q * q * 0.25 / (-p * p * p / 27)))
            else:
                phi = math.acos(math.sqrt(q * q * 0.25 / (-p * p * p / 27)))

            xc1 = 2 * tmp1 * math.cos(phi / 3) - a2 / 3
            xc2 = 2 * tmp1 * math.cos((phi + 2 * math.PI) / 3) - a2 / 3
            xc3 = 2 * tmp1 * math.cos((phi + 4 * math.PI) / 3) - a2 / 3

        returndistA = NewtonRaphson._distance(
            x1,
            y1,
            xc1,
            self.coeff[2] + self.coeff[1] * xc1 + self.coeff[0] * xc1 * xc1,
        )
        returndistB = NewtonRaphson._distance(
            x1,
            y1,
            xc2,
            self.coeff[2] + self.coeff[1] * xc2 + self.coeff[0] * xc2 * xc2,
        )
        returndistC = NewtonRaphson._distance(
            x1,
            y1,
            xc3,
            self.coeff[2] + self.coeff[1] * xc3 + self.coeff[0] * xc3 * xc3,
        )

        return math.min(returndistA, math.min(returndistB, returndistC))
