import sys
import warnings

import numpy  as np
import pandas as pd

from sklearn import model_selection, metrics

from cowboysmall.ml.classifiers.nb.nb       import GaussianNaiveBayes
from cowboysmall.ml.utilities.preprocessing import imbalanced
from cowboysmall.ml.utilities.metrics       import confusion_matrix


def main(argv):
    np.random.seed(1341)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    data = imbalanced.oversample(pd.read_csv('./data/csv/wine_white.csv', sep = ';'), 'quality')
    X = data.drop(['quality'], axis = 1).values
    Y = data.quality.values

    X, X_t, Y, Y_true = model_selection.train_test_split(X, Y, train_size = 0.67)

    nb = GaussianNaiveBayes()
    nb.fit(X, Y)
    Y_hat = nb.predict(X_t)

    print()
    print('Classification Experiment: White Wine')
    print()
    print()
    print()
    print()
    print('                   Result: {:.2f}% Correct'.format(100 * (Y_true == Y_hat).sum() / float(len(Y_true))))
    print()
    print('    Classification Report:')
    print()
    print(metrics.classification_report(Y_true, Y_hat))
    print()
    print('         Confusion Matrix:')
    print()
    print(confusion_matrix(Y_true, Y_hat))
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
