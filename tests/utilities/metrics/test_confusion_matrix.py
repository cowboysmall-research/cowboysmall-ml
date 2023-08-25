import unittest

import numpy as np

from cowboysmall.ml.utilities.metrics import confusion_matrix


class TestConfusionMatrix(unittest.TestCase):

    def test_confuion_matrix_two_outcomes_all_correct(self):
        cm = confusion_matrix([0, 1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 0, 1, 1, 0, 1])

        self.assertEqual(cm.shape[0], 2)
        self.assertEqual(cm.shape[1], 2)
        self.assertTrue((cm == np.array([[4, 0], [0, 4]])).all())


    def test_confuion_matrix_two_outcomes_some_correct(self):
        cm = confusion_matrix([0, 1, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 1, 0, 1])

        self.assertEqual(cm.shape[0], 2)
        self.assertEqual(cm.shape[1], 2)
        self.assertTrue((cm == np.array([[3, 1], [1, 3]])).all())


    def test_confuion_matrix_four_outcomes_all_correct(self):
        cm = confusion_matrix([0, 2, 3, 2, 1, 0, 1, 1, 3, 1, 2, 1, 2, 3], [0, 2, 3, 2, 1, 0, 1, 1, 3, 1, 2, 1, 2, 3])

        self.assertEqual(cm.shape[0], 4)
        self.assertEqual(cm.shape[1], 4)
        self.assertTrue((cm == np.array([[2, 0, 0, 0], [0, 5, 0, 0], [0, 0, 4, 0], [0, 0, 0, 3]])).all())


    def test_confuion_matrix_four_outcomes_some_correct(self):
        cm = confusion_matrix([0, 3, 3, 2, 1, 0, 1, 1, 3, 1, 2, 0, 2, 3], [0, 2, 3, 2, 1, 0, 1, 1, 3, 1, 2, 1, 2, 3])

        self.assertEqual(cm.shape[0], 4)
        self.assertEqual(cm.shape[1], 4)
        self.assertTrue((cm == np.array([[2, 1, 0, 0], [0, 4, 0, 0], [0, 0, 3, 0], [0, 0, 1, 3]])).all())



if __name__ == '__main__':
    unittest.main()
