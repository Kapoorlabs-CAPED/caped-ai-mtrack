import numpy as np


class LinearFunction:
    def __init__(self, points: np.ndarray):

        self.points = points
        self.coeff = []

    def fit(self):

        n_points = self.points.shape[0]
        delta = np.zeros(4)
        theta = np.zeros(2)
        for i in range(n_points):

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

    def distance(self, point):

        x1 = point[1]
        y1 = point[0]

        return abs(y1 - self.coeff[0] * x1 - self.coeff[1]) / np.sqrt(
            1 + self.coeff[0] * self.coeff[0]
        )
