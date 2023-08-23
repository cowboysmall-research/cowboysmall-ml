import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        lbls, cnts  = np.unique(y, return_counts = True)

        self.priors = np.array([value / float(y.shape[0]) for value in cnts])
        self.mean   = np.array([[X[y == i, j].mean() for j in range(X.shape[1])] for i in lbls])
        self.var    = np.array([[X[y == i, j].var(ddof = 1) for j in range(X.shape[1])] for i in lbls])


    def predict(self, X):
        return np.array([np.argmax(self.priors * np.prod(self.gaussian(X[i, :]), axis = 1)) for i in range(X.shape[0])])


    def gaussian(self, X):
        return np.exp(-np.square(X - self.mean) / (2 * self.var)) / np.sqrt(2 * np.pi * self.var)
