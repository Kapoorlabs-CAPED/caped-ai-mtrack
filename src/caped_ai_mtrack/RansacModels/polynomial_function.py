import numpy as np

from ..Solvers import NewtonRaphson
from .generalized_function import GeneralFunction


class PolynomialFunction(GeneralFunction):
    def __init__(self, points: list, degree: int):

        super().__init__(points, degree)
        self.points = np.asarray(self.points)
        self.num_points = self.get_num_points()

        self.coeff = np.zeros(degree)
        self.min_num_points = degree

    def fit(self):

        delta = np.zeros((self.degree, self.degree))
        tetha = np.zeros(self.degree)
        powCache = np.zeros(self.degree * self.degree)
        powCache[powCache.shape[0] - 1] = 1
        for i in range(self.points.shape[0]):

            point = self.points[i]
            y = point[0]
            x = point[1]

            power = 1
            for d in range(1, powCache.shape[0]):

                power *= x
                powCache[powCache.shape[0] - d - 1] = power

            for r in range(self.degree):
                for c in range(self.degree):
                    delta[r, c] += powCache[r + c]

            mulY = y

            for d in range(self.degree):

                tetha[self.degree - d - 1] += mulY
                mulY *= x
        deltainv = np.linalg.pinv(delta)
        for d in range(self.degree):

            for i in range(self.degree):
                self.coeff[d] += deltainv[d, i] * tetha[i]

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

        return NewtonRaphson(self.degree, self.coeff, x1, y1).run()

    def residuals(self, samples):

        shortest_distances = []
        num_points = len(samples)
        for i in range(num_points):

            point = samples[i]

            shortest_distances.append(self.distance(point))

        return shortest_distances
