import math
import warnings

import numpy as np

from .utils import check_consistent_length, clean_ransac, plot_ransac_gt


class Ransac:
    def __init__(
        self,
        data_points: list,
        model_class: type,
        degree: int,
        max_trials: int,
        iterations: int,
        residual_threshold: float,
        max_distance: float,
        min_samples: int = None,
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
        self.model_class = model_class
        self.degree = degree
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.max_distance = max_distance
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

        check_consistent_length(self.y, self.X)

        if self.min_samples is None:

            self.min_samples = self.X.shape[0] + 1
        elif 0 < self.min_samples < 1:
            self.min_samples = np.ceil(self.min_samples * self.X.shape[0])

    def _dynamic_max_trials(
        self, n_inliers, n_samples, min_samples, probability
    ):
        if n_inliers == 0:
            return np.inf

        if probability == 1:
            return np.inf

        if n_inliers == n_samples:
            return 1

        nom = math.log(1 - probability)
        denom = math.log(1 - (n_inliers / n_samples) ** min_samples)

        return int(np.ceil(nom / denom))

    def ransac(self, starting_points):

        if isinstance(starting_points, np.ndarray):
            starting_points = starting_points.tolist()

        print("starting_points", len(starting_points))
        y, X = zip(*starting_points)
        y = np.asarray(y)
        X = np.asarray(X)
        best_inlier_num = 0
        best_inlier_residuals_sum = np.inf
        best_inliers = []

        random_state = np.random.default_rng(self.random_state)

        # in case data is not pair of input and output, male it like it
        if not isinstance(starting_points, (tuple, list)):
            starting_points = (starting_points,)
        num_samples = len(starting_points)
        if not (0 < self.min_samples < num_samples):
            raise ValueError(
                f"`min_samples` must be in range (0, {num_samples})"
            )

        if self.residual_threshold < 0:
            raise ValueError("`residual_threshold` must be greater than zero")

        if self.max_trials < 0:
            raise ValueError("`max_trials` must be greater than zero")

        if not (0 <= self.stop_probability <= 1):
            raise ValueError("`stop_probability` must be in range [0, 1]")

        if (
            self.initial_inliers is not None
            and len(self.initial_inliers) != num_samples
        ):
            raise ValueError(
                f"RANSAC received a vector of initial inliers (length "
                f"{len(self.initial_inliers)}) that didn't match the number of "
                f"samples ({num_samples}). The vector of initial inliers should "
                f"have the same length as the number of samples and contain only "
                f"True (this sample is an initial inlier) and False (this one "
                f"isn't) values."
            )

        # for the first run use initial guess of inliers
        spl_idxs = (
            self.initial_inliers
            if self.initial_inliers is not None
            else random_state.choice(
                num_samples, self.min_samples, replace=False
            )
        )

        for num_trials in range(self.max_trials):
            # do sample selection according data pairs
            X_subset = X[spl_idxs]
            y_subset = y[spl_idxs]
            samples = [
                (y_subset[i], X_subset[i]) for i in range(y_subset.shape[0])
            ]
            # for next iteration choose random sample set and be sure that
            # no samples repeat
            spl_idxs = random_state.choice(
                num_samples, self.min_samples, replace=False
            )

            estimator = self.model_class(samples, self.degree)
            success = estimator.fit()
            # backwards compatibility
            if success is not None and not success:
                continue

            residuals = np.abs(estimator.residuals(starting_points))
            # consensus set / inliers
            inliers = residuals < self.residual_threshold
            residuals_sum = residuals.dot(residuals)

            # choose as new best model if number of inliers is maximal
            inliers_count = np.count_nonzero(inliers)
            if (
                # more inliers
                inliers_count > best_inlier_num
                # same number of inliers but less "error" in terms of residuals
                or (
                    inliers_count == best_inlier_num
                    and residuals_sum < best_inlier_residuals_sum
                )
            ):
                best_inlier_num = inliers_count
                best_inlier_residuals_sum = residuals_sum
                best_inliers = inliers
                dynamic_max_trials = self._dynamic_max_trials(
                    best_inlier_num,
                    num_samples,
                    self.min_samples,
                    self.stop_probability,
                )
                if (
                    best_inlier_num >= self.stop_sample_num
                    or best_inlier_residuals_sum <= self.stop_residuals_sum
                    or num_trials >= dynamic_max_trials
                ):
                    break

        # estimate final model using all inliers
        if any(best_inliers):
            # select inliers for each data array
            data_inliers_X = X[best_inliers]
            data_inliers_y = y[best_inliers]
            samples = [
                (data_inliers_y[i], data_inliers_X[i])
                for i in range(data_inliers_y.shape[0])
            ]
            estimator = self.model_class(samples, self.degree)
            estimator.fit()

        else:
            estimator = None
            best_inliers = None
            warnings.warn("No inliers found. Model not fitted")

        self.estimator_ = estimator
        self.inlier_mask_ = best_inliers
        return self.estimator_, self.inlier_mask_

    def extract_first_ransac_line(self, starting_points):

        ransac_result = self.ransac(starting_points)

        if ransac_result is not None:
            estimator, inliers = ransac_result
            results_inliers = []
            results_inliers_removed = []
            for i in range(0, len(starting_points)):
                if not inliers[i]:
                    # Not an inlier
                    results_inliers_removed.append(starting_points[i])
                    continue
                results_inliers.append(starting_points[i])
            return (
                np.array(results_inliers),
                np.array(results_inliers_removed),
                estimator,
            )

    def extract_multiple_lines(self):

        starting_points = np.asarray(self.data_points)
        data_points_list = np.copy(self.data_points)
        data_points_list = data_points_list.tolist()
        estimators = []
        estimator_inliers = []
        for index in range(0, self.iterations):

            if len(starting_points) <= self.min_samples:
                print(
                    "No more points available. Terminating search for RANSAC"
                )
                break
            ransac_first_line = self.extract_first_ransac_line(starting_points)
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

        segments = clean_ransac(estimators, estimator_inliers)
        yarray, xarray = zip(*data_points_list)
        plot_ransac_gt(segments, yarray, xarray, save_name=self.save_name)

        return estimators
