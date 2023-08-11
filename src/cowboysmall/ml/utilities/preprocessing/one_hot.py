import numpy as np


class OneHotEncoder(object):

    def __init__(self, data):
        self.items = list(set(data))
        self.enc = dict((item, i) for i, item in enumerate(self.items))
        self.dec = dict((i, item) for i, item in enumerate(self.items))
        self.dim = len(self.items)


    def encode(self, data):
        encoded = []

        for datum in data:
            encode = [0] * self.dim
            encode[self.enc[datum]] = 1
            encoded.append(encode)

        return np.array(encoded)


    def decode(self, data):
        return np.array([self.dec[datum.argmax()] for datum in data])
