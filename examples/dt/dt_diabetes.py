import sys
import warnings

import numpy  as np
import pandas as pd

from sklearn import model_selection, metrics

from cowboysmall.ml.classifiers.dt.tree import DecisionTree
from cowboysmall.ml.classifiers.dt.cost import entropy
from cowboysmall.ml.utilities.metrics   import confusion_matrix


def main(argv):
    np.random.seed(1340)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    data = pd.read_csv('./data/csv/diabetes.csv', names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y'])
    X = data.iloc[:, :8]
    Y = data.iloc[:, 8]

    X, X_t, Y, Y_t = model_selection.train_test_split(X, Y, train_size = 0.75)

    dt = DecisionTree(min_size = 20, cost = entropy)
    # dt = DecisionTree(min_size = 15, cost = gini)
    dt.fit(X, Y)
    P = dt.predict(X_t)

    print()
    print('Classification Experiment: Diabetes')
    print()
    print(dt)
    print()
    print()
    print()
    print('                   Result: {:.2f}% Correct'.format(100 * (Y_t == P).sum() / float(len(Y_t))))
    print()
    print('    Classification Report:')
    print()
    print(metrics.classification_report(Y_t, P))
    print()
    print('         Confusion Matrix:')
    print()
    print(confusion_matrix(Y_t, P))
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
