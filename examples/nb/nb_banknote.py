import sys
import warnings

import numpy as np

from ml.classifiers.nb.nb import GaussianNaiveBayes
from ml.utilities.metrics import confusion_matrix

from sklearn import model_selection, metrics


def main(argv):
    np.random.seed(1337)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)


    print()
    print('Classification Experiment: Banknote')
    print()


    data  = np.loadtxt('./data/csv/banknote.csv', delimiter = ',')
    X     = data[:, :4]
    Y     = data[:, 4].astype(int)


    X, X_t, Y, Y_true = model_selection.train_test_split(X, Y, train_size = 0.75)



    nb    = GaussianNaiveBayes()
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
    print(confusion_matrix(Y_true, Y_hat))
    print()



if __name__ == "__main__":
    main(sys.argv[1:])

