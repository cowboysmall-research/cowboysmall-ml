import sys
import warnings

import numpy as np
import pandas as pd

from sklearn import model_selection, metrics

import cowboysmall.ml.classifiers.nb.nb  as nb_01
import cowboysmall.ml.classifiers.nb.nb2 as nb_02

from cowboysmall.ml.utilities.metrics import confusion_matrix


def main(argv):
    np.random.seed(1024)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    pd.set_option("display.precision", 8)


    data = pd.read_csv('./data/csv/spambase.csv', header = None)

    X, X_t, Y, Y_t = model_selection.train_test_split(data, data.iloc[:, -1], train_size = 0.75)


    print()
    print('Classification Experiment: Spambase - Comparison of Classifiers')
    print()
    print()
    print()



    nb0 = nb_01.NaiveBayes()
    nb0.fit(X.iloc[:, :-1].values, Y.values)
    Y_hat1 = nb0.predict(X_t.iloc[:, :-1].values)

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


    nb1 = nb_02.NaiveBayes()
    nb1.fit(X, Y)
    Y_hat2 = nb1.predict(X_t.iloc[:, :-1])


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
            print('                    index: ', i)
            print('                  Regular: ', nb0.predict(X_t.iloc[i, :-1].values.reshape(1, X_t.shape[1] - 1))[0])
            print('               Vectorised: ', nb1.predict(pd.DataFrame(X_t.iloc[i, :-1].values.reshape(1, X_t.shape[1] - 1)))[0])
            print()
    else:
        print('           No Differences: ')
        print()


if __name__ == "__main__":
    main(sys.argv[1:])