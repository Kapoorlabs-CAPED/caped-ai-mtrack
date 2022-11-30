import numpy as np


class QuadraticFunction:
    def __init__(self, points: np.ndarray, degree: int):
        self.points = points
        self.degree = degree
        self.min_num_points = 3
        self.coeff = np.zeros(3)

    def fit(self):

        n_points = self.points.shape[0]
        delta = np.zeros(9)
        theta = np.zeros(3)
        for i in range(n_points):

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
