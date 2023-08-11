import numpy as np

from collections import Counter

from cowboysmall.ml.classifiers.dt.tree import DecisionTree
from cowboysmall.ml.classifiers.dt.cost import gini


class RandomForest:

    def __init__(self, max_depth = 10, min_size = 5, cost = gini, s_ratio = 1.0, dt_count = 1, f_count = None):
        self.trees   = [DecisionTree(max_depth, min_size, cost, f_count) for count in range(dt_count)]
        self.s_ratio = s_ratio


    def __str__(self):
        return '\n'.join('\nTree {}\n\n{}\n'.format(index + 1, tree) for index, tree in enumerate(self.trees))


    def fit(self, X, y):
        for tree in self.trees:
            samples = np.random.choice(np.arange(X.shape[0]), round(X.shape[0] * self.s_ratio), replace = False)
            tree.fit(X.iloc[samples, :], y.iloc[samples])


    def predict(self, X):
        return [Counter(row).most_common()[0][0] for row in np.column_stack([tree.predict(X) for tree in self.trees])]

