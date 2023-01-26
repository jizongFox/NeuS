from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def nerf():
    interval = 6 / 1000
    x = np.linspace(start=-3, stop=3, num=1000)

    def gaussian(x: np.ndarray, *, mean: float | int, std: float | int):
        return np.exp(-(x - mean) ** 2 / (2 * std ** 2)) / np.sqrt(2 * np.pi * std ** 2)

    v_density = gaussian(x, mean=0, std=1)
    # v_density = np.ones_like(v_density) / 6

    v_integral = np.cumsum(v_density * interval)
    T = np.exp(-v_integral)

    weight = v_density * T
    plt.figure()
    plt.plot(x, v_density)
    plt.figure()
    plt.plot(x, v_integral)

    plt.figure()
    plt.plot(x, weight)

    plt.show()
    print(weight.sum())


def neus():
    def sigmoid(x: np.ndarray, s: float = 1.0):
        return 1 / (1 + np.exp(-s * x))

    def get_p(x: np.ndarray, s: float = 1.0, *, distance_func, disable_negative=True):
        distance = distance_func(x)
        diff = sigmoid(distance[:-1], s) - sigmoid(distance[1:], s)
        denominator = sigmoid(distance, s)
        if disable_negative:
            return np.maximum(diff / denominator[:-1], 0)
        return diff / denominator[:-1]

    x = np.linspace(start=-30, stop=30, num=10000)

    def distance_func_complex(x):
        return np.maximum.reduce([-x, x - 7, -x + 8, x - 15])

    def distance_func_simple(x):
        return -x

    s = 10

    p_density = get_p(x, distance_func=distance_func_complex, s=s, disable_negative=True)

    T = np.cumprod((1 - p_density))
    weight = p_density * T

    x = x[:-1]

    plt.figure()
    plt.plot(x, p_density)
    plt.title("p_density")

    plt.figure()
    plt.plot(x, T)
    plt.title("T")

    plt.figure()
    plt.plot(x, weight)
    plt.title("weight")

    plt.show()
    print(weight.sum())


if __name__ == "__main__":
    neus()
