import numpy as np

from scipy.stats import norm

from collections import Counter


class AbstractNaiveBayes:

    def calculate_priors(self, c):
        raise NotImplementedError


    def calculate_likelihoods(self, X, y):
        likelihoods = {label:dict() for label in np.unique(y)}

        for label in likelihoods:
            for feature in range(X.shape[1]):
                likelihoods[label][feature] = self.calculate_priors(list(X[y == label, feature]))

        return likelihoods


    def calculate_posterior(self, X):
        posteriors = {}

        for label in self.priors:
            posterior = self.priors[label]

            for feature in range(X.shape[0]):
                likelihood = self.likelihoods[label][feature]
                posterior *= likelihood[X[feature]] if X[feature] in likelihood else 0

            posteriors[label] = posterior

        return max(posteriors, key = posteriors.get)


    def fit(self, X, y):
        self.priors      = self.calculate_priors(y)
        self.likelihoods = self.calculate_likelihoods(X, y)


    def predict(self, X):
        return [self.calculate_posterior(X[row]) for row in range(X.shape[0])]


class NaiveBayes(AbstractNaiveBayes):

    def calculate_priors(self, c):
        return {key:value / float(len(c)) for key, value in dict(Counter(c)).items()}


class GaussianNaiveBayes(AbstractNaiveBayes):

    def calculate_priors(self, c):
        g = norm(np.mean(c), np.std(c))
        return {i:g.pdf(i) for i in np.unique(c)}
