import numpy as np

from cowboysmall.ml.utilities.function.sigmoid import Tanh, Logistic
from cowboysmall.ml.utilities.function.activation import ReLU


class ConnectedLayer:

    def __init__(self, nodes, function = ReLU(), learning = 0.1, regular = 0, momentum = 0.1):
        self.nodes    = nodes
        self.function = function
        self.learning = learning
        self.regular  = regular
        self.momentum = momentum

    def create(self):
        self.weights = np.random.uniform(-1, 1, size = (self.nodes, self.next.get_nodes()))
        self.delta   = np.zeros((self.nodes, self.next.get_nodes()))

    def init(self, batch):
        self.batch      = batch
        self.activation = np.zeros((batch, self.nodes))
        self.bias       = np.zeros((batch, self.next.get_nodes()))

    def update(self):
        self.activation = self.function.f(self.prev.z())

    def set_next(self, next):
        self.next = next

    def set_prev(self, prev):
        self.prev = prev

    def get_nodes(self):
        return self.nodes

    def z(self):
        return (self.activation @ self.weights) + self.bias

    def update_parameters(self, delta):
        delta_w       = self.activation.T @ delta
        self.weights -= self.learning * ((delta_w / self.batch) + (self.regular * self.weights)) + (self.momentum * self.delta)
        self.bias    -= self.learning * (delta / self.batch)
        self.delta    = delta_w


class InputLayer(ConnectedLayer):

    def input(self, inputs):
        self.activation[:, :] = inputs


class HiddenLayer(ConnectedLayer):

    def __init__(self, nodes, function = Tanh(), **kwargs):
        ConnectedLayer.__init__(self, nodes, function, **kwargs)

    def calculate_delta(self, delta):
        return (delta @ self.weights.T) * self.function.f_prime(self.activation)


class OutputLayer(ConnectedLayer):

    def __init__(self, nodes, function = Logistic(), **kwargs):
        ConnectedLayer.__init__(self, nodes, function, **kwargs)

    def calculate_delta(self, labels):
        return (self.activation - labels) * self.function.f_prime(self.activation)

    def output(self):
        return self.activation
