#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_sample", type=int, default=10)
parser.add_argument("--theta0", type=float, default=1.0)
parser.add_argument("--theta1", type=float, default=1.0)
parser.add_argument("--theta2", type=float, default=0.0)
parser.add_argument("--theta3", type=float, default=0.0)
parser.add_argument("--sigma2", type=float, default=0.1)
args = parser.parse_args()

np.random.seed(args.seed)


def generate_sample(n_sample):
    x = np.linspace(0, 2 * np.pi, n_sample + 2)[1:-1].reshape((-1, 1))
    y = np.sin(x)
    t = np.sin(x) + np.random.normal(
        loc=0, scale=np.sqrt(args.sigma2), size=(n_sample, 1)
    )
    return x, y, t


def calc_kernel(xn, xm):
    diff_matrix = (
        np.repeat(xn, xm.shape[0], axis=1) - np.repeat(xm, xn.shape[0], axis=1).T
    )
    first_term = args.theta0 * np.exp(-args.theta1 / 2 * diff_matrix ** 2)

    second_term = args.theta2

    third_term = args.theta3 * np.dot(xn, xm.T)

    return first_term + second_term + third_term


class GaussianProcess:
    def __init__(self, x, t):
        self.x = x
        self.t = t

    def infer(self, xx):
        C = calc_kernel(self.x, self.x) + args.sigma2 * np.eye(self.x.shape[0])
        k = calc_kernel(self.x, xx)
        c = calc_kernel(xx, xx) + args.sigma2

        mu = np.dot(np.dot(k.T, np.linalg.inv(C)), self.t)[0][0]
        sigma2 = c - np.dot(np.dot(k.T, np.linalg.inv(C)), k)[0][0]

        return mu, sigma2


def main():
    x, _, t = generate_sample(args.n_sample)

    gp = GaussianProcess(x, t)

    x_infered = np.linspace(0, 2 * np.pi, 100)
    t_infered_upper = np.empty_like(x_infered)
    t_infered_mean = np.empty_like(x_infered)
    t_infered_lower = np.empty_like(x_infered)

    for i in range(x_infered.size):
        xx = x_infered[i].reshape((1, 1))
        mu, sigma2 = gp.infer(xx)

        t_infered_upper[i] = mu + np.sqrt(sigma2)
        t_infered_mean[i] = mu
        t_infered_lower[i] = mu - np.sqrt(sigma2)

    # =========
    # 以下描画処理

    plt.plot(x, t, "kx", label="data")

    plt.fill_between(
        x_infered, t_infered_lower, t_infered_upper, facecolor="grey", alpha=0.5
    )
    plt.plot(x_infered, t_infered_mean, "k", linestyle="dashed", label="inferred")

    x_true, y_true, _ = generate_sample(100)
    plt.plot(x_true, y_true, "k", label="true")

    plt.legend()

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()


if __name__ == "__main__":
    main()
