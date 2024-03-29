import sys

import numpy as np

from sklearn import model_selection, metrics

from cowboysmall.ml.classifiers.nb.nb import NaiveBayes
from cowboysmall.ml.utilities.metrics import confusion_matrix


def main(argv):
    np.random.seed(1024)

    data = np.loadtxt('./data/csv/spambase.csv', delimiter = ',')
    X = data[:, :57]
    Y = data[:, 57].astype(int)

    X, X_t, Y, Y_t = model_selection.train_test_split(X, Y, train_size = 0.75)

    nb = NaiveBayes()
    nb.fit(X, Y)
    Y_hat = nb.predict(X_t)

    print()
    print('Classification Experiment: Spambase')
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
