import sys
import warnings

import numpy as np
import pandas as pd

from sklearn import model_selection, metrics

from cowboysmall.ml.classifiers.nb.nb2 import NaiveBayes
from cowboysmall.ml.utilities.metrics  import confusion_matrix


def main(argv):
    np.random.seed(1337)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    data = pd.read_csv('./data/csv/diabetes.csv', header = None)

    X, X_t, Y, Y_t = model_selection.train_test_split(data, data.iloc[:, -1], train_size = 0.75)

    nb = NaiveBayes()
    nb.fit(X, Y)
    Y_hat = nb.predict(X_t.iloc[:, :-1])

    print()
    print('Classification Experiment: Diabetes')
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
