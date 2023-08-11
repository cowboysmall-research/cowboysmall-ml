import numpy as np


class ReLU:

    def f(self, x):
        return x * (x > 0)

    def f_prime(self, y):
        return (y > 0).astype(y.dtype)


class LeakyReLU:

    def f(self, x):
        return np.where(x > 0, x, 0.01 * x)

    def f_prime(self, y):
        return np.where(y > 0, 1, 0.01)
