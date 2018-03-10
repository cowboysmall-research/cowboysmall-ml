import sys
import warnings

import numpy  as np
import pandas as pd

from ml.classifiers.nb.nb import NaiveBayes

from sklearn import model_selection, metrics


def main(argv):
    np.random.seed(1337)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)


    print()
    print('Classification Experiment: White Wine')
    print()


    data  = pd.read_csv('./data/csv/wine_white.csv', sep = ';')
    X     = data.drop(['quality'], axis = 1).values
    Y     = data.quality.values - 3


    X, X_t, Y, Y_true = model_selection.train_test_split(X, Y, train_size = 0.75)



    nb    = NaiveBayes()
    nb.fit(X, Y)
    Y_hat = nb.predict(X_t)




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
    print(metrics.confusion_matrix(Y_true, Y_hat))
    print()



if __name__ == "__main__":
    main(sys.argv[1:])

