import numpy as np


class GeneralFunction:
    def __init__(self, points: np.ndarray):

        self.points = points

    def get_num_points(self):
        self.num_points = self.points.shape[0]
