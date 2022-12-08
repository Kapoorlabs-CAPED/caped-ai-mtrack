import matplotlib.pyplot as plt
import numpy as np
import pytest

from caped_ai_mtrack._tests.utils import plot_points, quadratic_points
from caped_ai_mtrack.Fits import Ransac
from caped_ai_mtrack.RansacModels import (
    LinearFunction,
    PolynomialFunction,
    QuadraticFunction,
)


@pytest.mark.parametrize("num_points", [250])
@pytest.mark.parametrize(
    "model", [LinearFunction, QuadraticFunction, PolynomialFunction]
)
@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("min_samples", [10, 20])
def test_ransac_models(num_points, min_samples, model, degree, save_name=""):

    plt.cla()
    pointlist = quadratic_points(num_points)
    yarray, xarray = zip(*pointlist)

    ransac_line = Ransac(
        pointlist,
        model,
        degree,
        min_samples=min_samples,
        max_trials=1000,
        iterations=3,
        residual_threshold=15,
        max_distance=10,
    )
    estimators = ransac_line.extract_multiple_lines()
    for estimator in estimators:

        ypredict = []
        for x in range(np.asarray(xarray).shape[0]):
            ypredict.append(estimator.predict(x))
        plot_points(plt, ypredict, yarray, xarray, save_name=save_name)


if __name__ == "__main__":

    # test_ransac_models(250, 2, LinearFunction, 2, save_name="_linear")
    # test_ransac_models(250, 3, QuadraticFunction, 3, save_name="_quadratic")
    test_ransac_models(250, 3, PolynomialFunction, 3, save_name="_polynomial")
