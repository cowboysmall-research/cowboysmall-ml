import numpy as np


class Logistic:

    def f(self, x):
        # return (np.exp(x) / (1 + np.exp(x))) if x.any() < 0 else (1 / (1 + np.exp(-x)))
        return 1 / (1 + np.exp(-x))

    def f_prime(self, y):
        return y * (1 - y)


class Tanh:

    def f(self, x):
        return np.tanh(x)

    def f_prime(self, y):
        return 1 - (y * y)
