import numpy as np


class Regression:
    def __init__(self, data_points: list, model_class: type, degree: int):

        self.data_points = data_points
        self.mode_class = model_class
        self.degree = degree

    def regression(self):

        # num_samples = len(self.data_points[0])

        self.model = self.model_class(self.data_points, self.degree)
        self.model.fit()
        np.abs(self.model.residuals())
