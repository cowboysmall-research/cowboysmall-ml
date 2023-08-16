import time

import numpy as np

from cowboysmall.ml.classifiers.mlp import print_layer_details, print_epoch_details


class Network:

    def __init__(self, verbose = True):
        self.verbose  = verbose
        self.layers   = []

    def add(self, layer):
        if self.layers:
            layer.set_prev(self.layers[-1])
            self.layers[-1].set_next(layer)
            self.layers[-1].create()

        self.layers.append(layer)

    def forward(self, features):
        for i in range(len(self.layers) - 1):
            self.layers[i].init(features.shape[0])

        self.layers[0].input(features)

        for i in range(1, len(self.layers)):
            self.layers[i].update()

    def backward(self, labels):
        delta = labels

        for i in reversed(range(1, len(self.layers))):
            delta = self.layers[i].calculate_delta(delta)
            self.layers[i - 1].update_parameters(delta)

    def cost(self, y, m):
        return np.sum((y - self.layers[-1].output()) ** 2) / (2 * m)

    def fit(self, X, y, batch = 100, epochs = 500):
        if self.verbose:
            print_layer_details(self.layers)

        m = X.shape[0]

        for i in range(1, epochs + 1):
            s = time.time()

            o = np.random.permutation(m)
            e = 0
            for j in range(0, m, batch):
                b  = o[j:j + batch]
                self.forward(X[b])
                self.backward(y[b])
                e += self.cost(y[b], len(b))

            d = time.time() - s

            if self.verbose and i % 10 == 0:
                print_epoch_details(i, epochs, batch, e, d)

    def predict(self, X):
        self.forward(X)

        return self.layers[-1].output()
