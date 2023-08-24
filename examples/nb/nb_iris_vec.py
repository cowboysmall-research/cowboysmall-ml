import sys

import numpy as np

from sklearn import model_selection, metrics

from cowboysmall.ml.classifiers.nb.nb2 import NaiveBayes
from cowboysmall.ml.utilities.metrics  import confusion_matrix
from cowboysmall.ml.utilities.preprocessing import LabelEncoder


def main(argv):
    np.random.seed(1024)

    data = np.loadtxt('./data/csv/iris_01.csv', delimiter = ',')
    X = data[:, :4]
    Y = data[:, 4].astype(int)

    le = LabelEncoder()
    le.fit(Y)

    X, X_t, Y, Y_t = model_selection.train_test_split(X, Y, train_size = 0.75)

    nb = NaiveBayes()
    nb.fit(X, le.transform(Y))
    Y_hat = le.inverse_transform(nb.predict(X_t))

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
