import numpy as np

from scipy.stats import norm


class GaussianLikelihood:

    def __init__(self, values):
        self.dist = norm(np.mean(values), np.std(values))

    def calculate(self, value):
        return self.dist.pdf(value)
