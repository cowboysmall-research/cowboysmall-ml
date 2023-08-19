import numpy as np

from collections import Counter


class NaiveBayes:

    def __init__(self, likelihood):
        self.likelihood = likelihood

    def fit(self, X, y):
        self.priors = {key: value / float(len(y)) for key, value in dict(Counter(y)).items()}
        self.likelihoods = {label: dict() for label in np.unique(y)}

        for label in self.likelihoods:
            for feature in range(X.shape[1]):
                self.likelihoods[label][feature] = self.likelihood(X[y == label, feature])

    def predict(self, X):
        predictions = []

        for row in X:
            posteriors = {}
            for label in self.priors:
                posterior = self.priors[label]
                for feature in range(row.shape[0]):
                    posterior *= self.likelihoods[label][feature].calculate(row[feature])
                posteriors[label] = posterior
            predictions.append(max(posteriors, key = posteriors.get))

        return predictions
