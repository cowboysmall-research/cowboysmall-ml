import numpy as np

from cowboysmall.ml.utilities.function.distribution import Gaussian


class NaiveBayes:

    def __init__(self, function = Gaussian()):
        self.function = function

    def fit(self, X, y):
        labels, counts = np.unique(y, return_counts = True)

        self.priors = np.array([value / float(y.shape[0]) for value in counts])
        self.mean   = np.array([[X[y == i, j].mean() for j in range(X.shape[1])] for i in labels])
        self.var    = np.array([[X[y == i, j].var()  for j in range(X.shape[1])] for i in labels])

    def predict(self, X):
        return np.array([np.argmax(self.posterior(X[row])) for row in range(X.shape[0])])

    def posterior(self, X):
        posteriors = []

        for label, prior in enumerate(self.priors):
            posterior = prior

            for feature in range(X.shape[0]):
                posterior *= self.function.f(X[feature], self.mean[label, feature], self.var[label, feature])

            posteriors.append(posterior)

        return posteriors
