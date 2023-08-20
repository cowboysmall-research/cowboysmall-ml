import sys
import warnings

import numpy as np
import pandas as pd

from sklearn import model_selection, metrics

from cowboysmall.ml.classifiers.nb.nb2 import NaiveBayes
from cowboysmall.ml.utilities.metrics  import confusion_matrix


def main(argv):
    np.random.seed(1024)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    cols = ['sl', 'sw', 'pl', 'pw', 'c']
    data = pd.read_csv('./data/csv/iris_01.csv', sep = ',', names = cols)

    X, X_t, Y, Y_t = model_selection.train_test_split(data, data['c'], train_size = 0.75)

    nb = NaiveBayes()
    nb.fit(X, Y, cols)
    Y_hat = nb.predict(X_t.drop(['c'], axis = 1).values) + 1

    print()
    print('Classification Experiment: Iris')
    print()
    print()
    print()
    print()
    print('                   Result: {:.2f}% Correct'.format(100 * (Y_t == Y_hat).sum() / float(len(Y_t))))
    print()
    print('    Classification Report:')
    print()
    print(metrics.classification_report(Y_t, Y_hat))
    print()
    print('         Confusion Matrix:')
    print()
    print(confusion_matrix(Y_t, Y_hat))
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
