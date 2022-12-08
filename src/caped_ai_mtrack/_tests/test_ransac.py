import matplotlib.pyplot as plt
import numpy as np
import pytest

from caped_ai_mtrack._tests.utils import (
    plot_points,
    quadratic_points,
    random_points,
)
from caped_ai_mtrack.Fits import Ransac
from caped_ai_mtrack.RansacModels import LinearFunction, QuadraticFunction


@pytest.mark.parametrize("num_points", [250])
@pytest.mark.parametrize("model", [LinearFunction, QuadraticFunction])
@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("min_samples", [10, 20])
def quadratic_points_ransac(
    num_points, min_samples, model, degree, save_name=""
):

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


@pytest.mark.parametrize("num_points", [250])
@pytest.mark.parametrize("model", [LinearFunction, QuadraticFunction])
@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("min_samples", [10, 20])
def linear_points_ransac(num_points, min_samples, model, degree, save_name=""):

    plt.cla()
    pointlist = random_points(num_points)
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

    quadratic_points_ransac(
        250, 2, LinearFunction, 2, save_name="_linear_quadratic"
    )
    plt.cla()
    quadratic_points_ransac(
        250, 3, QuadraticFunction, 3, save_name="_quadratic_quadratic"
    )

    plt.cla()
    linear_points_ransac(250, 2, LinearFunction, 2, save_name="_linear_linear")
    plt.cla()
    linear_points_ransac(
        250, 3, QuadraticFunction, 3, save_name="_quadratic_linear"
    )
