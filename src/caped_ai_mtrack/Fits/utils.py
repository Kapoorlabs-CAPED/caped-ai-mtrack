import numbers
import os

import matplotlib.pyplot as plt
import numpy as np


def root_dir():
    return os.path.dirname(os.path.abspath(__file__))


def clean_ransac(estimators, estimator_inliers):

    segments = []

    for i in range(len(estimators)):

        estimator = estimators[i]
        estimator_inlier = estimator_inliers[i]
        estimator_inliers_list = np.copy(estimator_inlier)
        yarray, xarray = zip(*estimator_inliers_list.tolist())
        yarray = np.asarray(yarray)
        xarray = np.asarray(xarray)

        ypredict = []
        xpredict = []
        for j in range(np.asarray(xarray).shape[0]):
            x = xarray[j]
            y = estimator.predict(x)
            ypredict.append(y)
            xpredict.append(x)
        segments.append([ypredict, xpredict])
    return segments


def plot_ransac_gt(segments, yarray, xarray, save_name=""):

    plt.cla()
    plt.plot(xarray, yarray)
    for ypredict, xpredict in segments:
        plt.plot(xpredict, ypredict)
    plt.title("MTrack Ransac")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig(root_dir() + save_name)


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(length) for length in lengths]
        )


def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection."
                % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )
