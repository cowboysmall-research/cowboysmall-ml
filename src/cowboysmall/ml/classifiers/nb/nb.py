import numpy as np

from scipy.stats import norm

from collections import Counter


class GaussianNaiveBayes:

    def calculate_priors(self, c):
        return {key: value / float(len(c)) for key, value in dict(Counter(c)).items()}

    def calculate_stats(self, X, y):
        stats = {label:dict() for label in np.unique(y)}

        for label in stats:
            for feature in range(X.shape[1]):
                c = X[y == label, feature]
                stats[label][feature] = (np.mean(c), np.std(c))

        return stats

    def calculate_posterior(self, X):
        posteriors = {}

        for label in self.priors:
            posterior = self.priors[label]

            for feature in range(X.shape[0]):
                posterior *= norm(*self.stats[label][feature]).pdf(X[feature])

            posteriors[label] = posterior

        return posteriors

    def argmax(self, posteriors):
        return max(posteriors, key = posteriors.get)

    def fit(self, X, y):
        self.priors = self.calculate_priors(y)
        self.stats  = self.calculate_stats(X, y)

    def predict(self, X):
        return [self.argmax(self.calculate_posterior(row)) for row in X]
