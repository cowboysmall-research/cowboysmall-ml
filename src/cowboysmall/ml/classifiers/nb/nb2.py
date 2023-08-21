import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        self.priors = y.value_counts() / y.shape[0]
        self.mean   = X.groupby(X.keys()[-1])[X.keys()[:-1]].mean()
        self.var    = X.groupby(X.keys()[-1])[X.keys()[:-1]].var()


    def predict(self, X):
        posteriors = np.array([self.priors * np.prod(self.gaussian(X.iloc[i, :]), axis = 1) for i in range(X.shape[0])])
        return np.array([np.argmax(posterior) for posterior in posteriors])


    def gaussian(self, X):
        return np.exp(-np.square(X - self.mean) / (2 * self.var)) / np.sqrt(2 * np.pi * self.var)
