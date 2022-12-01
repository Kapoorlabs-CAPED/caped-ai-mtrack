import pytest

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
