import numpy as np


class NaiveBayes:

    def fit(self, X, y, cols):
        self.priors = np.array(y.value_counts()) / y.shape[0]
        self.mean   = X.groupby(cols[-1])[cols[:-1]].mean()
        self.var    = X.groupby(cols[-1])[cols[:-1]].var()

    def predict(self, X):
        return np.array([np.argmax(self.priors * np.prod(self.gaussian(row), axis = 1)) for row in X])

    def gaussian(self, X):
        return np.exp(-np.square(X - self.mean) / (2 * self.var)) / np.sqrt(2 * (np.pi) * self.var)
