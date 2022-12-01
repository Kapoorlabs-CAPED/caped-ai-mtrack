import os

import matplotlib.pyplot as plt


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


def plot_points(pointlist: list):

    yarray, xarray = zip(*pointlist)
    plt.plot(xarray, yarray, "ro")
    plt.title("Example kymograph")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig(root_dir())


def root_dir():
    return os.path.dirname(os.path.abspath(__file__))
