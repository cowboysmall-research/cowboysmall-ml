import sys

import numpy as np

from sklearn import model_selection, metrics

import cowboysmall.ml.classifiers.nb.nb  as nb_01
import cowboysmall.ml.classifiers.nb.nb2 as nb_02

from cowboysmall.ml.utilities.metrics import confusion_matrix


def main(argv):
    np.random.seed(1024)

    data = np.loadtxt('./data/csv/spambase.csv', delimiter = ',')
    X = data[:, :57]
    Y = data[:, 57].astype(int)

    X, X_t, Y, Y_t = model_selection.train_test_split(X, Y, train_size = 0.75)


    print()
    print('Classification Experiment: Spambase - Comparison of Classifiers')
    print()


    nb1 = nb_01.NaiveBayes()
    nb1.fit(X, Y)
    Y_hat1 = nb1.predict(X_t)

    print()
    print('       Spambase - Regular:')
    print()
    print('                   Result: {:.2f}% Correct'.format(100 * (Y_t == Y_hat1).sum() / float(len(Y_t))))
    print()
    print('    Classification Report:')
    print()
    print(metrics.classification_report(Y_t, Y_hat1))
    print()
    print('         Confusion Matrix:')
    print()
    print(confusion_matrix(Y_t, Y_hat1))
    print()


    nb2 = nb_02.NaiveBayes()
    nb2.fit(X, Y)
    Y_hat2 = nb2.predict(X_t)

    print()
    print('    Spambase - Vectorised:')
    print()
    print('                   Result: {:.2f}% Correct'.format(100 * (Y_t == Y_hat2).sum() / float(len(Y_t))))
    print()
    print('    Classification Report:')
    print()
    print(metrics.classification_report(Y_t, Y_hat2))
    print()
    print('         Confusion Matrix:')
    print()
    print(confusion_matrix(Y_t, Y_hat2))
    print()


    print()
    print('Classification Experiment: Spambase - Differences in Classifiers')
    print()

    differences = []
    for i in range(len(Y_hat1)):
        if Y_hat1[i] != Y_hat2[i]:
            differences.append(i)

    if differences:
        for i in differences:
            print()
            print('                    index: ', i)
            print('                  Regular: ', nb1.predict(X_t[i, :].reshape(1, X_t.shape[1]))[0])
            print('               Vectorised: ', nb2.predict(X_t[i, :].reshape(1, X_t.shape[1]))[0])
            print()
    else:
        print()
        print('           No Differences: ')
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
