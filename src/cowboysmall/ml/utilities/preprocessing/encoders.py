import numpy as np

from abc import ABC, abstractmethod


class Encoder(ABC):

    def fit(self, data):
        items = np.unique(data)

        self.enc = {item: i for i, item in enumerate(items)}
        self.dec = {i: item for i, item in enumerate(items)}
        self.dim = len(items)

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
        encoded = [0] * self.dim
        encoded[self.enc[datum]] = 1
        return encoded

    def decode(self, datum):
        return self.dec[datum.argmax()]


class LabelEncoder(Encoder):

    def encode(self, datum):
        return self.enc[datum]

    def decode(self, datum):
        return self.dec[datum]
