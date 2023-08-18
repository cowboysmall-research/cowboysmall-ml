import numpy as np

from cowboysmall.ml.utilities.function.sigmoid import Logistic
from cowboysmall.ml.classifiers.glm import print_details


class LogisticRegression:

    def __init__(self, learning = 0.1, verbose = True):
        self.verbose = verbose
        self.logistic = Logistic()

    def add_intercept(self, X):
        x = np.ones((X.shape[0], X.shape[1] + 1))
        x[:, 1:] = X
        return x

    def hypothesis(self, X):
        return self.logistic.f(X @ self.theta)

    def cost(self, X, y):
        h = self.hypothesis(X)
        return -np.sum((y * np.log(h)) + ((1 - y) * np.log(1 - h))) / X.shape[0]

    def gradient(self, X, y):
        return (X.T @ (self.hypothesis(X) - y)) / X.shape[0]

    def hessian(self, X):
        h = self.hypothesis(X)
        return (X.T @ np.diag(h[:, 0]) @ np.diag(1 - h[:, 0]) @ X) / X.shape[0]

    def fit(self, X, y, tolerance = 0.00001, iterations = 50):
        X = self.add_intercept(X)
        y = y.reshape(y.shape[0], 1)

        self.theta = np.zeros((X.shape[1], 1))

        J = [self.cost(X, y)]
        i = 1
        while i <= iterations:
            G = self.gradient(X, y)
            H = self.hessian(X)

            if np.linalg.det(H) == 0:
                break

            self.theta -= np.linalg.solve(H, G)

            J_new = self.cost(X, y)
            if np.isnan(J_new) or abs(J[-1] - J_new) < tolerance:
                break

            J.append(J_new)
            i += 1

        if self.verbose:
            print_details(i, J[-1], self.theta)

    def predict(self, X):
        return self.hypothesis(self.add_intercept(X))[:, 0]
