import numpy as np

from ..RansacModels import LinearFunction, QuadraticFunction
from .ransac import Ransac
from .utils import check_consistent_length, clean_estimators


class ComboRansac(Ransac):
    def __init__(
        self,
        data_points: list,
        model_linear: LinearFunction,
        model_quadratic: QuadraticFunction,
        min_samples: int,
        max_trials: bool,
        iterations: int,
        residual_threshold: float,
        timeindex: int = 0,
        stop_probability: float = 1,
        stop_sample_num: float = np.inf,
        max_skips: float = np.inf,
        stop_n_inliers: float = np.inf,
        stop_residuals_sum: int = 0,
        stop_score: float = np.inf,
        random_state=None,
        initial_inliers=None,
        save_name="",
    ):

        self.data_points = data_points
        self.model_linear = model_linear
        self.model_quadratic = model_quadratic
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.timeindex = timeindex
        self.iterations = iterations
        self.stop_probability = stop_probability
        self.stop_sample_num = stop_sample_num
        self.stop_n_inliers = stop_n_inliers
        self.max_skips = max_skips
        self.stop_residuals_sum = stop_residuals_sum
        self.random_state = random_state
        self.stop_score = stop_score
        self.initial_inliers = initial_inliers
        self.save_name = save_name
        y, X = zip(*self.data_points)
        self.y = np.asarray(y)
        self.X = np.asarray(X)
        self.degree = 2
        if self.min_samples is None:

            self.min_samples = self.X.shape[0] + 1
        elif 0 < self.min_samples < 1:
            self.min_samples = np.ceil(self.min_samples * self.X.shape[0])

        self.ransac_line = Ransac(
            self.data_points,
            self.model_linear,
            2,
            self.min_samples,
            self.max_trials,
            self.iterations,
            self.residual_threshold,
            self.timeindex,
            self.stop_probability,
            self.stop_sample_num,
            self.max_skips,
            self.stop_n_inliers,
            self.stop_residuals_sum,
            self.stop_score,
            self.random_state,
            self.initial_inliers,
            self.save_name,
        )

        check_consistent_length(self.y, self.X)

        self.ransac_quadratic = Ransac(
            self.data_points,
            self.model_quadratic,
            3,
            self.min_samples,
            self.max_trials,
            self.iterations,
            self.residual_threshold,
            self.timeindex,
            self.stop_probability,
            self.stop_sample_num,
            self.max_skips,
            self.stop_n_inliers,
            self.stop_residuals_sum,
            self.stop_score,
            self.random_state,
            self.initial_inliers,
            self.save_name,
        )

    def extract_multiple_lines(self):

        starting_points = np.asarray(self.data_points)

        data_points_list = np.copy(self.data_points)
        data_points_list = data_points_list.tolist()
        estimators = []
        estimator_inliers = []
        self.orig_min_samples = self.min_samples
        if self.min_samples < 3:

            self.min_samples = 3
        for index in range(0, self.iterations):

            if len(starting_points) <= self.min_samples:
                print(
                    "No more points available. Terminating search for RANSAC"
                )
                break
            ransac_first_quad = (
                self.ransac_quadratic.extract_first_ransac_line(
                    starting_points
                )
            )
            if ransac_first_quad is not None:
                (
                    inlier_points,
                    inliers_removed_from_starting,
                    estimator,
                ) = ransac_first_quad

            else:
                starting_points = []
            estimators.append(estimator)
            estimator_inliers.append(inlier_points)
            if len(starting_points) < self.min_samples:
                print(
                    "Not sufficeint inliers found %d , threshold=%d, therefore halting"
                    % (len(starting_points), self.min_samples)
                )

                break
            starting_points = inliers_removed_from_starting

        # segments = clean_ransac(estimators, estimator_inliers)
        # yarray, xarray = zip(*data_points_list)
        # plot_ransac_gt(segments, yarray, xarray, save_name=self.save_name)
        estimator_inliers = [
            item for sublist in estimator_inliers for item in sublist
        ]
        starting_points = np.asarray(estimator_inliers)

        data_points_list = np.copy(estimator_inliers)
        data_points_list = data_points_list.tolist()
        estimators = []
        estimator_inliers = []
        self.min_samples = self.orig_min_samples
        for index in range(0, self.iterations):

            if len(starting_points) <= self.min_samples:
                print(
                    "No more points available. Terminating search for RANSAC"
                )
                break
            ransac_first_line = self.ransac_line.extract_first_ransac_line(
                starting_points
            )
            if ransac_first_line is not None:
                (
                    inlier_points,
                    inliers_removed_from_starting,
                    estimator,
                ) = ransac_first_line

            else:
                starting_points = []
            estimators.append(estimator)
            estimator_inliers.append(inlier_points)
            if len(starting_points) < self.min_samples:
                print(
                    "Not sufficeint inliers found %d , threshold=%d, therefore halting"
                    % (len(starting_points), self.min_samples)
                )

                break
            starting_points = inliers_removed_from_starting

        # segments = clean_ransac(estimators, estimator_inliers)
        # yarray, xarray = zip(*data_points_list)
        # plot_ransac_gt(segments, yarray, xarray, save_name=self.save_name)
        estimators, estimator_inliers = clean_estimators(
            estimators=estimators,
            estimator_inliers=estimator_inliers,
            degree=self.degree,
            timeindex=self.timeindex,
        )

        return estimators, estimator_inliers
