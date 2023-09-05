
import numpy as np


class Gaussian:

    def f(self, X, mean = 0, var = 1):
        return np.exp(-np.square(X - mean) / (2 * var)) / np.sqrt(2 * np.pi * var)

    def f_prime(self, y, mean = 0, var = 1):
        return (-y * self.f(y, mean, var)) / var
