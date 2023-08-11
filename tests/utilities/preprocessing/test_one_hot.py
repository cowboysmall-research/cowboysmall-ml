import unittest

import numpy as np

from cowboysmall.ml.utilities.preprocessing import one_hot


class TestOneHot(unittest.TestCase):

    def test_one_hot_encode_empty(self):
        data    = np.array([2, 4, 3, 4, 3, 4, 5, 4, 5, 6])
        encoder = one_hot.OneHotEncoder(data)

        encoded = encoder.encode(np.array([]))

        self.assertEqual(len(encoded), 0)


    def test_one_hot_encode_not_empty(self):
        data    = np.array([2, 4, 3, 4, 3, 4, 5, 4, 5, 6])
        encoder = one_hot.OneHotEncoder(data)

        encoded = encoder.encode(np.array([2, 3, 4, 5, 6]))

        self.assertEqual(len(encoded), 5)
        self.assertEqual(encoded[0].tolist(), [1, 0, 0, 0, 0])
        self.assertEqual(encoded[1].tolist(), [0, 1, 0, 0, 0])
        self.assertEqual(encoded[2].tolist(), [0, 0, 1, 0, 0])
        self.assertEqual(encoded[3].tolist(), [0, 0, 0, 1, 0])
        self.assertEqual(encoded[4].tolist(), [0, 0, 0, 0, 1])


    def test_one_hot_decode_empty(self):
        data    = np.array([2, 4, 3, 4, 3, 4, 5, 4, 5, 6])
        encoder = one_hot.OneHotEncoder(data)

        decoded = encoder.decode(np.array([]))

        self.assertEqual(len(decoded), 0)


    def test_one_hot_decode_not_empty(self):
        data    = np.array([2, 4, 3, 4, 3, 4, 5, 4, 5, 6])
        encoder = one_hot.OneHotEncoder(data)

        encoded = np.array([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0]])
        decoded = encoder.decode(encoded)

        self.assertEqual(len(decoded), 2)
        self.assertEqual(decoded[0], 5)
        self.assertEqual(decoded[1], 3)


if __name__ == '__main__':
    unittest.main()
