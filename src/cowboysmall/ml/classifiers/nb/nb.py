import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        labels, counts = np.unique(y, return_counts = True)

        self.labels = labels
        self.priors = {value[0]: value[1] / float(y.shape[0]) for value in zip(labels, counts)}
        self.stats  = {label: dict() for label in labels}

        for label in labels:
            for feature in range(X.shape[1]):
                self.stats[label][feature] = (X[y == label, feature].mean(), X[y == label, feature].var())


    def predict(self, X):
        predictions = np.empty(X.shape[0])

        for i in range(X.shape[0]):
            pmax = -1
            lmax = None

            for label in self.labels:
                posterior = self.priors[label]

                for feature in range(X[i].shape[0]):
                    posterior *= self.gaussian(X[i, feature], self.stats[label][feature])

                if posterior > pmax:
                    pmax = posterior
                    lmax = label

            predictions[i] = lmax

        return predictions


    def gaussian(self, x, stats):
        return np.exp(-np.square(x - stats[0]) / (2 * stats[1])) / np.sqrt(2 * np.pi * stats[1])
