import numpy as np

from scipy.stats import norm

from collections import Counter


class GaussianNaiveBayes:

    def priors(self, c):
        return {key: value / float(len(c)) for key, value in dict(Counter(c)).items()}

    def distributions(self, X, y):
        distributions = {label: dict() for label in np.unique(y)}

        for label in distributions:
            for feature in range(X.shape[1]):
                c = X[y == label, feature]
                distributions[label][feature] = norm(np.mean(c), np.std(c))

        return distributions

    def posteriors(self, row):
        posteriors = {}

        for label in self.priors:
            posterior = self.priors[label]

            for feature in range(row.shape[0]):
                posterior *= self.dists[label][feature].pdf(row[feature])

            posteriors[label] = posterior

        return posteriors

    def argmax(self, posteriors):
        return max(posteriors, key = posteriors.get)

    def fit(self, X, y):
        self.priors = self.priors(y)
        self.dists  = self.distributions(X, y)

    def predict(self, X):
        return [self.argmax(self.posteriors(row)) for row in X]
