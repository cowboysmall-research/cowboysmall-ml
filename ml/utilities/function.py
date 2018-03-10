
import numpy as np


class Identity:

    def f(self, x):
        return x

    def f_prime(self, y):
        return y



class Sigmoid:

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def f_prime(self, y):
        return y * (1 - y)



class Tanh:

    def f(self, x):
        return np.tanh(x)

    def f_prime(self, y):
        return 1 - (y * y)



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

