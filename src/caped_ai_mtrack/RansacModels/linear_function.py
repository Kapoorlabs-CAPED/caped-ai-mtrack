import numpy as np

from .generalized_function import GeneralFunction


class LinearFunction(GeneralFunction):
    def __init__(self, points: list):

        super(GeneralFunction, self).__init__(points, 1)
        self.points = np.asarray(self.points)
        self.num_points = self.get_num_points()
        self.min_num_points = 2
        self.coeff = np.zeros(self.min_num_points)

    def fit(self):

        delta = np.zeros(4)
        theta = np.zeros(2)
        for i in range(self.num_points):

            point = self.points[i]

            y = point[0]
            x = point[1]

            xx = x * x
            xy = x * y

            delta[0] += xx
            delta[1] += x
            delta[2] += x
            delta[3] += 1

            theta[0] += xy
            theta[1] += y

        delta = np.invert(delta)

        self.coeff[0] = delta[0] * theta[0] + delta[1] * theta[1]
        self.coeff[1] = delta[2] * theta[0] + delta[3] * theta[1]

        return True

    def get_coefficients(self, j):

        return self.coeff[j]

    def predict(self, x):

        y = self.coeff[0] * x + self.coeff[1]

        return y

    def distance(self, point):

        x1 = point[1]
        y1 = point[0]

        return abs(y1 - self.coeff[0] * x1 - self.coeff[1]) / np.sqrt(
            1 + self.coeff[0] * self.coeff[0]
        )

    def residuals(self):

        shortest_distances = []

        for i in range(self.num_points):

            point = self.points[i]

            shortest_distances.append(self.distance(point))

        return shortest_distances
