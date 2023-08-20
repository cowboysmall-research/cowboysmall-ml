import numpy as np

from collections import Counter


class NaiveBayes:

    def fit(self, X, y):
        self.priors = {key: value / float(len(y)) for key, value in dict(Counter(y)).items()}
        self.stats  = {label: dict() for label in np.unique(y)}

        for label in self.stats:
            for feature in range(X.shape[1]):
                self.stats[label][feature] = (np.mean(X[y == label, feature]), np.var(X[y == label, feature]))

    def predict(self, X):
        predictions = []

        for row in X:
            posteriors = {}
            for label in self.priors:
                posterior = self.priors[label]
                for feature in range(row.shape[0]):
                    posterior *= self.gaussian(row[feature], self.stats[label][feature])
                posteriors[label] = posterior
            predictions.append(max(posteriors, key = posteriors.get))

        return np.array(predictions)

    def gaussian(self, x, stats):
        return np.exp(-np.square(x - stats[0]) / (2 * stats[1])) / np.sqrt(2 * (np.pi) * stats[1])
