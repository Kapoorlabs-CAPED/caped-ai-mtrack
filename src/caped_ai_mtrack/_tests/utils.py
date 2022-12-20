import os

import numpy as np


def quadratic_points(num_points: int):

    pointlist = []

    for i in range(num_points):
        if i < num_points // 4:
            pointlist.append((i * i, i))
        if i >= num_points // 4 and i < num_points // 2:
            pointlist.append((0.1 * i * i, i))

        if i >= num_points // 2 and i < 3 * num_points / 4:
            pointlist.append((-i + 200, i))
        if i >= 3 * num_points // 4 and i <= num_points:
            pointlist.append((i // 2, i))

    return pointlist


def random_points(num_points: int):

    pointlist = []

    for i in range(num_points):
        if i < num_points // 4:
            pointlist.append((i, i))
        if i >= num_points // 4 and i < num_points // 2:
            pointlist.append((2 * i - 60, i))
        if i >= num_points // 2 and i < 3 * num_points / 4:
            pointlist.append((-i + 200, i))
        if i >= 3 * num_points // 4 and i <= num_points:
            pointlist.append((i // 2, i))

    return pointlist


def plot_points(
    plt,
    yarray: np.ndarray,
    ypred: np.ndarray,
    xarray: np.ndarray,
    save_name="",
):

    plt.plot(xarray, yarray, xarray, ypred)
    plt.title("MTrack Ransac")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig(root_dir() + save_name)


def root_dir():
    return os.path.dirname(os.path.abspath(__file__))
