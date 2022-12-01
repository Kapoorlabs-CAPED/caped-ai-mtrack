import numpy as np

from ..Solvers import NewtonRaphson
from .generalized_function import GeneralFunction


class PolynomialFunction(GeneralFunction):
    def __init__(self, points: np.ndarray, degree: int):

        super(GeneralFunction, self).__init__(points)
        self.num_points = self.get_num_points()

        self.degree = degree
        self.coeff = np.zeros(degree + 1)
        self.min_num_points = degree + 1

    def fit(self):

        vandermonde = np.zeros((self.points, self.degree + 1))
        y = np.zeros(self.num_points)
        i = 0
        for k in range(self.num_points):
            point = self.points[k]
            x = point[1]
            for j in range(self.degree + 1):
                vandermonde[i, j] = x**j
            y[i] = point[0]
            i = i + 1

        X = np.copy(vandermonde)
        Y = np.zeros((self.num_points, self.num_points))
        for k in range(Y.shape[0]):
            Y[k, :] = y

        Q, R = np.linalg.qr(X)
        self.coefficients = np.flip(np.linalg.inv(R).dot(Q.T.dot(y)))

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

        return NewtonRaphson(self.degree, self.coeff).run(x1, y1)
