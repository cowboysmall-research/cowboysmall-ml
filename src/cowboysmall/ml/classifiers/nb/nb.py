import numpy as np

from collections import Counter


class NaiveBayes:

    def fit(self, X, y):
        self.priors = {key: float(value) / len(y) for key, value in dict(Counter(y)).items()}
        self.stats  = {label: dict() for label in np.unique(y)}

        for label in self.stats:
            for feature in range(X.shape[1]):
                self.stats[label][feature] = (X[y == label, feature].mean(), X[y == label, feature].var(ddof = 1))

    def predict(self, X):
        predictions = np.empty(X.shape[0], dtype = np.int64)

        for i in range(X.shape[0]):
            posteriors = {}
            for label in self.priors:
                posterior = self.priors[label]
                for feature in range(X[i].shape[0]):
                    posterior *= self.gaussian(X[i][feature], self.stats[label][feature])
                posteriors[label] = posterior

            predictions[i] = 0 if len(set(posteriors.values())) == 1 else max(posteriors, key = posteriors.get)

        return predictions

    def gaussian(self, x, stats):
        return np.exp(-np.square(x - stats[0]) / (2 * stats[1])) / np.sqrt(2 * np.pi * stats[1])
