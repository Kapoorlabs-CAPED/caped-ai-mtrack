import matplotlib.pyplot as plt
import numpy as np
import pytest

from caped_ai_mtrack._tests.utils import plot_points, random_points
from caped_ai_mtrack.Fits import Ransac
from caped_ai_mtrack.RansacModels import LinearFunction, PolynomialFunction


@pytest.mark.parametrize("num_points", [250])
@pytest.mark.parametrize("model", [LinearFunction])
@pytest.mark.parametrize("degree", [1])
@pytest.mark.parametrize("min_samples", [10, 20])
def test_ransac_models(num_points, min_samples, model, degree):

    pointlist = random_points(num_points)
    yarray, xarray = zip(*pointlist)
    plot_points(plt, yarray, xarray, "original", color="ro")
    ransac_line = Ransac(
        pointlist,
        model,
        degree,
        min_samples=min_samples,
        max_trials=1000,
        iterations=20,
        residual_threshold=5,
        max_distance=10,
    )
    estimators = ransac_line.extract_multiple_lines()
    if degree > 2:
        for estimator in estimators:
            print(
                estimator.get_coefficients(0),
                "* x * x",
                "+",
                estimator.get_coefficients(1),
                "  * x",
                estimator.get_coefficients(2),
                "+",
                "c",
            )
    else:
        for estimator in estimators:
            print(
                estimator.get_coefficients(0),
                "* x",
                "+",
                estimator.get_coefficients(1),
                "+",
                "c",
            )

        ypredict = []
        for x in range(np.asarray(xarray).shape[0]):
            ypredict.append(estimator.predict(x))
        plot_points(plt, ypredict, xarray, "predictions", color="go")


if __name__ == "__main__":

    test_ransac_models(250, 20, PolynomialFunction, 3)
