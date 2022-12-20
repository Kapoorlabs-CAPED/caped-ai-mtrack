import matplotlib.pyplot as plt
import numpy as np
import pytest

from caped_ai_mtrack._tests.utils import (
    plot_points,
    quadratic_points,
    random_points,
    root_dir,
)
from caped_ai_mtrack.Fits import Ransac
from caped_ai_mtrack.RansacModels import LinearFunction, QuadraticFunction


@pytest.mark.parametrize("num_points", [250])
@pytest.mark.parametrize("model", [LinearFunction, QuadraticFunction])
@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("min_samples", [2, 2])
def test_quadratic_points_ransac(
    num_points, min_samples, model, degree, save_name=""
):

    plt.cla()
    pointlist = quadratic_points(num_points)
    yarray, xarray = zip(*pointlist)

    ransac_quad = Ransac(
        pointlist,
        model,
        degree,
        min_samples=min_samples,
        max_trials=10000,
        iterations=10,
        residual_threshold=0.01,
        max_distance=10,
        save_name=save_name,
    )
    estimators = ransac_quad.extract_multiple_lines()
    for estimator in estimators:

        ypredict = []
        for x in range(np.asarray(xarray).shape[0]):
            ypredict.append(estimator.predict(x))
        plot_points(plt, ypredict, yarray, xarray, save_name=save_name)


@pytest.mark.parametrize("num_points", [250])
@pytest.mark.parametrize("model", [LinearFunction, QuadraticFunction])
@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("min_samples", [2, 3])
def test_linear_points_ransac(
    num_points, min_samples, model, degree, save_name=""
):

    plt.cla()
    pointlist = random_points(num_points)
    yarray, xarray = zip(*pointlist)

    ransac_line = Ransac(
        pointlist,
        model,
        degree,
        min_samples=min_samples,
        max_trials=10000,
        iterations=10,
        residual_threshold=0.01,
        max_distance=10,
        save_name=save_name,
    )
    estimators = ransac_line.extract_multiple_lines()
    for estimator in estimators:

        ypredict = []
        for x in range(np.asarray(xarray).shape[0]):
            ypredict.append(estimator.predict(x))
        plot_points(plt, ypredict, yarray, xarray, save_name=save_name)


@pytest.mark.parametrize("num_points", [250])
def test_draw_linear_points(num_points):

    plt.cla()
    pointlist = random_points(num_points)
    yarray, xarray = zip(*pointlist)
    plt.plot(xarray, yarray)
    plt.title("GT")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig(root_dir() + "GT")


@pytest.mark.parametrize("num_points", [250])
def test_draw_quad_points(num_points):

    plt.cla()
    pointlist = quadratic_points(num_points)
    yarray, xarray = zip(*pointlist)
    plt.plot(xarray, yarray)
    plt.title("GT")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig(root_dir() + "GT_quad")


if __name__ == "__main__":

    plt.cla()
    test_draw_quad_points(250)
    plt.cla()
    test_linear_points_ransac(
        250, 2, LinearFunction, 3, save_name="_linear_linear"
    )
    plt.cla()
    test_quadratic_points_ransac(
        250, 3, QuadraticFunction, 3, save_name="_quadratic_linear"
    )
