import numpy as np

from cowboysmall.ml.classifiers.dt.node import Node, LeafNode
from cowboysmall.ml.classifiers.dt.cost import gini


class DecisionTree:

    def __init__(self, max_depth = 10, min_size = 5, cost = gini, f_count = None):
        self.root      = None
        self.max_depth = max_depth
        self.min_size  = min_size
        self.cost      = cost
        self.f_count   = f_count


    def __str__(self):
        return self.root.to_string()


    def build_tree(self, X, y, features, depth = 1):
        best_cost     = 999999
        best_criteria = None
        best_split    = None
        best_labels   = None

        for feature in features:

            for value in np.unique(X.loc[:, feature]):
                if isinstance(value, int) or isinstance(value, float):
                    left, right = X.loc[:, feature] <  value, X.loc[:, feature] >= value
                else:
                    left, right = X.loc[:, feature] == value, X.loc[:, feature] != value

                y_left, y_right = y[left], y[right]

                if y_left.shape[0] * y_right.shape[0] > 0:
                    cost = self.cost([y_left, y_right], np.unique(y))

                    if best_cost > cost:
                        best_cost     = cost
                        best_criteria = (feature, value)
                        best_split    = (X.loc[left, :], X.loc[right, :])
                        best_labels   = (y_left, y_right)

        if not best_labels or depth > self.max_depth or len(best_labels[0]) < self.min_size or len(best_labels[1]) < self.min_size:
            return LeafNode(y)
        else:
            return Node(
                best_criteria, 
                self.build_tree(best_split[0], best_labels[0], features, depth + 1), 
                self.build_tree(best_split[1], best_labels[1], features, depth + 1)
            )


    def fit(self, X, y):
        if self.f_count:
            features = np.random.choice(list(X), self.f_count, replace = False)
        else:
            features = list(X)

        self.root = self.build_tree(X, y, features)


    def predict(self, X):
        return [self.root.predict(row) for _, row in X.iterrows()]
            
