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
        j = np.searchsorted(self.items, datum)
        return np.array([i == j for i in range(len(self.items))], dtype = np.int8)

    def decode(self, datum):
        return self.items[datum.argmax()]


class LabelEncoder(Encoder):

    def encode(self, datum):
        return np.searchsorted(self.items, datum)

    def decode(self, datum):
        return self.items[datum]
