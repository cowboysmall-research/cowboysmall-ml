import sys
import warnings

import numpy  as np
import pandas as pd

from sklearn import model_selection, metrics

from cowboysmall.ml.classifiers.dt.tree     import DecisionTree
from cowboysmall.ml.classifiers.dt.cost     import entropy
from cowboysmall.ml.utilities.preprocessing import imbalanced
from cowboysmall.ml.utilities.metrics       import confusion_matrix


def main(argv):
    np.random.seed(1337)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    data = imbalanced.oversample(pd.read_csv('./data/csv/wine_red.csv', sep = ';'), 'quality')
    X = data.iloc[:, :11]
    Y = data.iloc[:, 11]

    X, X_t, Y, Y_t = model_selection.train_test_split(X, Y, train_size = 0.5)

    dt = DecisionTree(cost = entropy)
    # dt = DecisionTree(cost = gini)
    dt.fit(X, Y)
    P = dt.predict(X_t)

    print()
    print('Classification Experiment: Red Wine')
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
