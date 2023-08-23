import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        labels, counts = np.unique(y, return_counts = True)

        self.priors = np.array([value / float(y.shape[0]) for value in counts])
        self.mean   = np.array([[X[y == i, j].mean() for j in range(X.shape[1])] for i in labels])
        self.var    = np.array([[X[y == i, j].var()  for j in range(X.shape[1])] for i in labels])


    def predict(self, X):
        return np.apply_along_axis(np.argmax, 1, np.apply_along_axis(self.posterior, 1, X))


    def posterior(self, X):
        return self.priors * np.prod(self.gaussian(X), axis = 1)


    def gaussian(self, X):
        return np.exp(-np.square(X - self.mean) / (2 * self.var)) / np.sqrt(2 * np.pi * self.var)
