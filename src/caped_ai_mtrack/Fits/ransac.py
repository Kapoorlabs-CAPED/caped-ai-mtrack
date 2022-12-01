import math

import numpy as np
from RansacModels import GeneralFunction


class Ransac:
    def __init__(
        self,
        data_points: list,
        model_class: GeneralFunction,
        degree: int,
        min_samples: int,
        max_trials: int,
        iterations: int,
        residual_threshold: float,
        max_distance: float,
        stop_probability: float = 1,
        stop_sample_num: float = np.inf,
        stop_residuals_sum: int = 0,
        random_state=None,
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
        self.stop_residuals_sum = stop_residuals_sum
        self.random_state = random_state

    def _dynamic_max_trials(
        self, n_inliers, n_samples, min_samples, probability
    ):
        """Determine number trials such that at least one outlier-free subset is
        sampled for the given inlier/outlier ratio.

        Parameters
        ----------
        n_inliers : int
                Number of inliers in the data.
        n_samples : int
                Total number of samples in the data.
        min_samples : int
                Minimum number of samples chosen randomly from original data.
        probability : float
                Probability (confidence) that one outlier-free sample is generated.

        Returns
        -------
        trials : int
                Number of trials.
        """
        if n_inliers == 0:
            return np.inf

        if probability == 1:
            return np.inf

        if n_inliers == n_samples:
            return 1

        nom = math.log(1 - probability)
        denom = math.log(1 - (n_inliers / n_samples) ** min_samples)

        return int(np.ceil(nom / denom))

    def ransac(self):

        best_inlier_num = 0
        best_inlier_residuals_sum = np.inf
        best_inliers = []

        random_state = np.random.default_rng(self.random_state)

        num_samples = len(self.data_points[0])

        if not (0 < self.min_samples < num_samples):
            raise ValueError(
                f"`min_samples` must be in range (0, {num_samples})"
            )

        if self.residual_threshold < 0:
            raise ValueError("`residual_threshold` must be greater than zero")

        if self.max_trials < 0:
            raise ValueError("`max_trials` must be greater than zero")

        # for the first run use initial guess of inliers
        spl_idxs = random_state.choice(
            num_samples, self.min_samples, replace=False
        )

        # estimate model for current random sample set

        for num_trials in range(self.max_trials):
            # do sample selection according data pairs
            samples = [d[spl_idxs] for d in self.data_points]

            # for next iteration choose random sample set and be sure that
            # no samples repeat
            spl_idxs = random_state.choice(
                num_samples, self.min_samples, replace=False
            )

            self.model = self.model_class(samples, self.degree)
            success = self.model.fit()
            # backwards compatibility
            if success is not None and not success:
                continue

            self.model = self.model_class(self.data_points, self.degree)
            residuals = np.abs(self.model.residuals())
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
            data_inliers = [d[best_inliers] for d in self.data_points]
            self.model = self.model_class(data_inliers, self.degree)

            self.model.fit()

        else:
            best_inliers = None

        # Define the actual ransac algo
        return inliers

    def extract_first_ransac_line(self):

        inliers = self.ransac()

        results_inliers = []
        results_inliers_removed = []
        for i in range(0, len(self.data_points)):
            if inliers[i] is False:
                # Not an inlier
                results_inliers_removed.append(self.data_points[i])
                continue
            x = self.data_points[i][1]
            y = self.data_points[i][0]
            results_inliers.append((x, y))
        return np.array(results_inliers), np.array(results_inliers_removed)

    def extract_multiple_lines_and_save(self):

        starting_points = self.data_points
        for index in range(0, self.iterations):
            if len(starting_points) <= self.min_samples:
                print(
                    "No more points available. Terminating search for RANSAC"
                )
                break
            (
                inlier_points,
                inliers_removed_from_starting,
            ) = self.extract_first_ransac_line()
            if len(inlier_points) < self.min_samples:
                print(
                    "Not sufficeint inliers found %d , threshold=%d, therefore halting"
                    % (len(inlier_points), self.min_samples)
                )
                break
            starting_points = inliers_removed_from_starting
