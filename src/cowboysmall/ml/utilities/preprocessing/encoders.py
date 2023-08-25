import numpy as np

from abc import ABC, abstractmethod


class Encoder(ABC):

    def fit(self, data):
        self.items = np.unique(data)

    def transform(self, data):
        return np.array([self.encode(datum) for datum in data])

    def inverse_transform(self, data):
        return np.array([self.decode(datum) for datum in data])

    @abstractmethod
    def encode(self, datum):
        pass

    @abstractmethod
    def decode(self, datum):
        pass


class OneHotEncoder(Encoder):

    def encode(self, datum):
        encoded = [0] * len(self.items)
        encoded[np.searchsorted(self.items, datum)] = 1
        return encoded

    def decode(self, datum):
        return self.items[datum.argmax()]


class LabelEncoder(Encoder):

    def encode(self, datum):
        return np.searchsorted(self.items, datum)

    def decode(self, datum):
        return self.items[datum]
