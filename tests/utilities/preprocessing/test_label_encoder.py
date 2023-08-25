import unittest

import numpy as np

from cowboysmall.ml.utilities.preprocessing import LabelEncoder


class TestLabelEncoder(unittest.TestCase):

    def test_label_encode_empty(self):
        data    = np.array([2, 4, 3, 4, 3, 4, 5, 4, 5, 6])
        encoder = LabelEncoder()
        encoder.fit(data)

        encoded = encoder.transform(np.array([]))

        self.assertEqual(len(encoded), 0)


    def test_label_encode_not_empty(self):
        data    = np.array([2, 4, 3, 4, 3, 4, 5, 4, 5, 6])
        encoder = LabelEncoder()
        encoder.fit(data)

        encoded = encoder.transform(np.array([2, 3, 4, 5, 6]))

        self.assertEqual(len(encoded), 5)
        self.assertEqual(encoded[0].tolist(), 0)
        self.assertEqual(encoded[1].tolist(), 1)
        self.assertEqual(encoded[2].tolist(), 2)
        self.assertEqual(encoded[3].tolist(), 3)
        self.assertEqual(encoded[4].tolist(), 4)


    def test_label_decode_empty(self):
        data    = np.array([2, 4, 3, 4, 3, 4, 5, 4, 5, 6])
        encoder = LabelEncoder()
        encoder.fit(data)

        decoded = encoder.inverse_transform(np.array([]))

        self.assertEqual(len(decoded), 0)


    def test_label_decode_not_empty(self):
        data    = np.array([2, 4, 3, 4, 3, 4, 5, 4, 5, 6])
        encoder = LabelEncoder()
        encoder.fit(data)

        encoded = np.array([3, 1])
        decoded = encoder.inverse_transform(encoded)

        self.assertEqual(len(decoded), 2)
        self.assertEqual(decoded[0], 5)
        self.assertEqual(decoded[1], 3)


if __name__ == '__main__':
    unittest.main()
