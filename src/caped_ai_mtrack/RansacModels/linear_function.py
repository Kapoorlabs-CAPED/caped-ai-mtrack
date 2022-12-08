import numpy as np

from .generalized_function import GeneralFunction


class LinearFunction(GeneralFunction):
    def __init__(self, points: list, degree: int = 1):

        super().__init__(points, degree)
        self.points = points
        self.num_points = self.get_num_points()
        self.min_num_points = 2
        self.coeff = np.zeros(self.min_num_points)
        self.degree = degree

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

        delta = np.linalg.pinv(np.reshape(delta, (2, 2)))

        self.coeff[0] = delta[0, 0] * theta[0] + delta[0, 1] * theta[1]
        self.coeff[1] = delta[1, 0] * theta[0] + delta[1, 1] * theta[1]

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

    def residuals(self, samples):

        shortest_distances = []
        num_points = len(samples)
        for i in range(num_points):

            point = samples[i]

            shortest_distances.append(self.distance(point))

        return shortest_distances
