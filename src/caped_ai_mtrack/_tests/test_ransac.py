import pytest

from ..Fits import Ransac
from ..RansacModels import (
    LinearFunction,
    PolynomialFunction,
    QuadraticFunction,
)
from .utils import plot_points, random_points


@pytest.mark.parametrize("num_points", [250])
@pytest.mark.parametrize(
    "model", [LinearFunction, PolynomialFunction, QuadraticFunction]
)
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_ransac_models(num_points, model, degree):

    pointlist = random_points(num_points)
    plot_points(pointlist)

    ransac_line = Ransac(
        pointlist,
        LinearFunction,
        1,
        max_trials=10,
        iterations=10,
        residual_threshold=2,
        max_distance=2,
    )
    ransac_line.extract_multiple_lines()
