import pytest

from caped_ai_mtrack._tests.utils import plot_points, random_points
from caped_ai_mtrack.Fits import Ransac
from caped_ai_mtrack.RansacModels import LinearFunction


@pytest.mark.parametrize("num_points", [250])
@pytest.mark.parametrize("model", [LinearFunction])
@pytest.mark.parametrize("degree", [1])
@pytest.mark.parametrize("min_samples", [10, 20])
def test_ransac_models(num_points, min_samples, model, degree):

    pointlist = random_points(num_points)
    plot_points(pointlist)

    ransac_line = Ransac(
        pointlist,
        LinearFunction,
        1,
        min_samples=min_samples,
        max_trials=100,
        iterations=20,
        residual_threshold=1,
        max_distance=1,
    )
    estimators = ransac_line.extract_multiple_lines()
    for estimator in estimators:
        print(
            estimator.get_coefficients(0),
            "* x",
            "+",
            estimator.get_coefficients(1),
            " ",
            "c",
        )


if __name__ == "__main__":

    test_ransac_models(250, 20, LinearFunction, 1)
