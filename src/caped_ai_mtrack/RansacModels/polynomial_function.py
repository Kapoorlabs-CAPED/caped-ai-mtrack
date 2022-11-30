import numpy as np

from ..Solvers import NewtonRaphson


class PolynomialFunction:
    def __init__(self, points: np.ndarray, degree: int):

        self.points = points
        self.degree = degree
        self.coeff = np.zeros(degree + 1)
        self.min_num_points = degree + 1

    def fit(self):

        n_points = self.points.shape[0]
        vandermonde = np.zeros((self.points, self.degree + 1))
        y = np.zeros(n_points)
        i = 0
        for k in range(n_points):
            point = self.points[k]
            x = point[1]
            for j in range(self.degree + 1):
                vandermonde[i, j] = x**j
            y[i] = point[0]
            i = i + 1

        X = np.copy(vandermonde)
        Y = np.zeros((n_points, n_points))
        for k in range(Y.shape[0]):
            Y[k, :] = y

        Q, R = np.linalg.qr(X)
        self.coefficients = np.flip(np.linalg.inv(R).dot(Q.T.dot(y)))

    def distance(self, point):

        x1 = point[1]
        y1 = point[0]

        return NewtonRaphson(self.degree, self.coeff).run(x1, y1)
