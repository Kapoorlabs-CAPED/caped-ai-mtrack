import math

import numpy as np

from .generalized_function import GeneralFunction

_epsilon = 1.0e-15


class QuadraticFunction(GeneralFunction):
    def __init__(self, points: list, degree: int):

        super().__init__(points, degree)
        self.points = np.asarray(self.points)
        self.num_points = self.get_num_points()
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

        delta = np.linalg.pinv(np.reshape(delta, (3, 3)))

        self.coeff[0] = (
            delta[0, 0] * theta[0]
            + delta[0, 1] * theta[1]
            + delta[0, 2] * theta[2]
        )
        self.coeff[1] = (
            delta[1, 0] * theta[0]
            + delta[1, 1] * theta[1]
            + delta[1, 2] * theta[2]
        )
        self.coeff[2] = (
            delta[2, 0] * theta[0]
            + delta[2, 1] * theta[1]
            + delta[2, 2] * theta[2]
        )

        return True

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
        a2 = np.true_divide(3 * self.coeff[1] * self.coeff[0], a3 + _epsilon)
        a1 = np.true_divide(
            2 * self.coeff[2] * self.coeff[0]
            - 2 * self.coeff[0] * y1
            + 1
            + self.coeff[1] * self.coeff[1],
            a3 + _epsilon,
        )
        a0 = np.true_divide(
            self.coeff[2] * self.coeff[1] - y1 * self.coeff[1] - x1,
            a3 + _epsilon,
        )

        p = (3 * a1 - a2 * a2) / 3
        q = (-9 * a1 * a2 + 27 * a0 + 2 * a2 * a2 * a2) / 27

        tmp1 = np.sqrt(abs(-p) / 3)
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
            xc2 = 2 * tmp1 * math.cos((phi + 2 * math.pi) / 3) - a2 / 3
            xc3 = 2 * tmp1 * math.cos((phi + 4 * math.pi) / 3) - a2 / 3

        returndistA = self._distance(
            x1,
            y1,
            xc1,
            self.coeff[2] + self.coeff[1] * xc1 + self.coeff[0] * xc1 * xc1,
        )
        returndistB = self._distance(
            x1,
            y1,
            xc2,
            self.coeff[2] + self.coeff[1] * xc2 + self.coeff[0] * xc2 * xc2,
        )
        returndistC = self._distance(
            x1,
            y1,
            xc3,
            self.coeff[2] + self.coeff[1] * xc3 + self.coeff[0] * xc3 * xc3,
        )

        return min(returndistA, min(returndistB, returndistC))

    def _distance(self, minx, miny, maxx, maxy):

        distance = (maxx - minx) * (maxx - minx) + (maxy - miny) * (
            maxy - miny
        )

        return np.sqrt(distance)

    def residuals(self, samples):

        shortest_distances = []
        num_points = len(samples)
        for i in range(num_points):

            point = samples[i]

            shortest_distances.append(self.distance(point))

        return shortest_distances
