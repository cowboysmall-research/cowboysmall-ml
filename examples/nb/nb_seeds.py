import sys
import warnings

import numpy as np

from sklearn import model_selection, metrics, preprocessing

from cowboysmall.ml.classifiers.nb.nb import NaiveBayes
from cowboysmall.ml.utilities.metrics import confusion_matrix


def main(argv):
    np.random.seed(1066)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    data = np.loadtxt('./data/csv/seeds.csv', delimiter = ',')
    X = data[:, :7]
    Y = data[:, 7].astype(int)

    le = preprocessing.LabelEncoder()
    le.fit(Y)

    X, X_t, Y, Y_true = model_selection.train_test_split(X, Y, train_size = 0.75)

    nb = NaiveBayes()
    nb.fit(X, le.transform(Y))
    Y_hat = le.inverse_transform(nb.predict(X_t))

    print()
    print('Classification Experiment: Seeds')
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
