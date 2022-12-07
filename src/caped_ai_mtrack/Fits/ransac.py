import warnings

import numpy as np

from .utils import check_consistent_length

_EPSILON = np.spacing(1)


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
        is_data_valid: bool = None,
        stop_probability: float = 1,
        stop_sample_num: float = np.inf,
        max_skips: float = np.inf,
        stop_n_inliers: float = np.inf,
        stop_residuals_sum: int = 0,
        stop_score: float = np.inf,
        random_state=None,
    ):

        self.data_points = data_points
        self.model_class = model_class
        self.degree = degree
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.max_distance = max_distance
        self.is_data_valid = is_data_valid
        self.iterations = iterations
        self.stop_probability = stop_probability
        self.stop_sample_num = stop_sample_num
        self.stop_n_inliers = stop_n_inliers
        self.max_skips = max_skips
        self.stop_residuals_sum = stop_residuals_sum
        self.random_state = random_state
        self.stop_score = stop_score

        self._sortlist()
        y, X = zip(*self.data_points)
        self.y = np.asarray(y)
        self.X = np.asarray(X)

        check_consistent_length(self.X, self.y)

        if self.min_samples is None:

            self.min_samples = self.X.shape[0] + 1
        elif 0 < self.min_samples < 1:
            self.min_samples = np.ceil(self.min_samples * self.X.shape[0])

    def _dynamic_max_trials(
        self, n_inliers, n_samples, min_samples, probability
    ):
        inlier_ratio = n_inliers / float(n_samples)
        nom = max(_EPSILON, 1 - probability)
        denom = max(_EPSILON, 1 - inlier_ratio**min_samples)
        if nom == 1:
            return 0
        if denom == 1:
            return float("inf")
        return abs(float(np.ceil(np.log(nom) / np.log(denom))))

    def _sortlist(self):

        self.data_points = sorted(self.data_points, key=lambda x: x[1])

    def ransac(self, starting_points):

        if isinstance(starting_points, np.ndarray):
            starting_points = starting_points.tolist()

        print(len(starting_points))
        y, X = zip(*starting_points)
        self.y = np.asarray(y)
        self.X = np.asarray(X)

        if self.stop_probability < 0 or self.stop_probability > 1:
            raise ValueError("`stop_probability` must be in range [0, 1].")

        if self.residual_threshold is None:
            # MAD (median absolute deviation)
            residual_threshold = np.median(np.abs(self.y - np.median(self.y)))
        else:
            residual_threshold = self.residual_threshold

        n_inliers_best = 1
        score_best = -np.inf
        inlier_mask_best = None
        X_inlier_best = None
        y_inlier_best = None

        # number of data samples
        n_samples = self.X.shape[0]
        sample_idxs = np.arange(n_samples)
        self.n_skips_no_inliers_ = 0
        self.n_skips_invalid_data_ = 0
        self.n_skips_invalid_model_ = 0
        self.n_trials_ = 0
        max_trials = self.max_trials
        while self.n_trials_ < max_trials:
            self.n_trials_ += 1

            if (
                self.n_skips_no_inliers_
                + self.n_skips_invalid_data_
                + self.n_skips_invalid_model_
            ) > self.max_skips:
                break

            # choose random sample set
            random_state = np.random.default_rng(self.random_state)
            subset_idxs = random_state.choice(
                n_samples, self.min_samples, replace=False
            )

            X_subset = self.X[subset_idxs]
            y_subset = self.y[subset_idxs]

            # check if random sample set is valid
            if self.is_data_valid is not None and not self.is_data_valid(
                X_subset, y_subset
            ):
                self.n_skips_invalid_data_ += 1
                continue
            samples = [
                (y_subset[i], X_subset[i]) for i in range(y_subset.shape[0])
            ]

            # fit model for current random sample set
            estimator = self.model_class(samples, self.degree)
            success = estimator.fit()
            residuals_subset = np.abs(estimator.residuals())
            # check if estimated model is valid
            if success is not None and not success:
                self.n_skips_invalid_model_ += 1
                continue

            estimator = self.model_class(starting_points, self.degree)
            success = estimator.fit()
            residuals_subset = np.abs(estimator.residuals())
            # classify data into inliers and outliers
            inlier_mask_subset = residuals_subset <= residual_threshold

            n_inliers_subset = np.sum(inlier_mask_subset)

            # less inliers -> skip current random sample
            if n_inliers_subset < n_inliers_best:
                self.n_skips_no_inliers_ += 1
                continue

            # extract inlier data set
            inlier_idxs_subset = sample_idxs[inlier_mask_subset]
            X_inlier_subset = self.X[inlier_idxs_subset]
            y_inlier_subset = self.y[inlier_idxs_subset]

            # score of inlier data set
            score_subset = np.sum(residuals_subset) / (len(residuals_subset))
            # same number of inliers but worse score -> skip current random
            # sample
            if (
                n_inliers_subset == n_inliers_best
                and score_subset < score_best
            ):
                continue

            # save current random sample as best sample
            n_inliers_best = n_inliers_subset
            score_best = score_subset
            inlier_mask_best = inlier_mask_subset
            X_inlier_best = X_inlier_subset
            y_inlier_best = y_inlier_subset

            max_trials = min(
                max_trials,
                self._dynamic_max_trials(
                    n_inliers_best,
                    n_samples,
                    self.min_samples,
                    self.stop_probability,
                ),
            )

            # break if sufficient number of inliers or score is reached
            if (
                n_inliers_best >= self.stop_n_inliers
                or score_best >= self.stop_score
            ):
                break

        # if none of the iterations met the required criteria
        if inlier_mask_best is None:
            if (
                self.n_skips_no_inliers_
                + self.n_skips_invalid_data_
                + self.n_skips_invalid_model_
            ) > self.max_skips:
                raise ValueError(
                    "RANSAC skipped more iterations than `max_skips` without"
                    " finding a valid consensus set. Iterations were skipped"
                    " because each randomly chosen sub-sample failed the"
                    " passing criteria. See estimator attributes for"
                    " diagnostics (n_skips*)."
                )
            else:
                raise ValueError(
                    "RANSAC could not find a valid consensus set. All"
                    " `max_trials` iterations were skipped because each"
                    " randomly chosen sub-sample failed the passing criteria."
                    " See estimator attributes for diagnostics (n_skips*)."
                )
        else:
            if (
                self.n_skips_no_inliers_
                + self.n_skips_invalid_data_
                + self.n_skips_invalid_model_
            ) > self.max_skips:
                warnings.warn(
                    "RANSAC found a valid consensus set but exited"
                    " early due to skipping more iterations than"
                    " `max_skips`. See estimator attributes for"
                    " diagnostics (n_skips*)."
                )

        # estimate final model using all inliers
        samples = [
            (y_inlier_best[i], X_inlier_best[i])
            for i in range(y_inlier_best.shape[0])
        ]
        estimator = self.model_class(samples, self.degree)
        estimator.fit()
        self.estimator_ = estimator
        self.inlier_mask_ = inlier_mask_best
        return self.inlier_mask_, self.estimator_

    def extract_first_ransac_line(self, starting_points):

        inliers, estimator = self.ransac(starting_points)

        results_inliers = []
        results_inliers_removed = []
        for i in range(0, len(starting_points)):
            if not inliers[i]:
                # Not an inlier
                results_inliers_removed.append(starting_points[i])
                continue
            x = starting_points[i][1]
            y = starting_points[i][0]
            results_inliers.append((x, y))

        return (
            np.array(results_inliers),
            np.array(results_inliers_removed),
            estimator,
        )

    def extract_multiple_lines(self):

        starting_points = self.data_points
        estimators = []
        for index in range(0, self.iterations):

            if len(starting_points) <= self.min_samples:
                print(
                    "No more points available. Terminating search for RANSAC"
                )
                break
            (
                inlier_points,
                inliers_removed_from_starting,
                estimator,
            ) = self.extract_first_ransac_line(starting_points)
            estimators.append(estimator)
            if len(starting_points) < self.min_samples:
                print(
                    "Not sufficeint inliers found %d , threshold=%d, therefore halting"
                    % (len(starting_points), self.min_samples)
                )

                break
            starting_points = inliers_removed_from_starting

        return estimators
