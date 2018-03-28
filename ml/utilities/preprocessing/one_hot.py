
import numpy as np


def forward(Y):
    Y_min = np.min(Y)
    Y_max = np.max(Y)
    Y_dim = Y_max - Y_min + 1
    Y_n   = np.array([[0] * Y_dim for _ in range(Y.shape[0])])

    for y_n, y in zip(Y_n, Y):
        y_n[y - Y_min] = 1

    return Y_n


def reverse(Y, Y_min = 0):
    return np.array([y.argmax() + Y_min for y in Y])

