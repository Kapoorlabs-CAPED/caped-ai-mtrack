class GeneralFunction:
    def __init__(self, points: list, degree: int):

        self.points = points
        self.degree = degree

    def get_num_points(self):
        self.num_points = len(self.points)
        return self.num_points

    def fit(self):
        pass

    def get_coefficients(self, j):
        pass

    def predict(self, x):
        pass

    def distance(self, point):
        pass

    def residuals(self):
        pass
