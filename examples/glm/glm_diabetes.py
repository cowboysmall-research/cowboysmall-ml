import sys
import warnings

import numpy as np

from ml.classifiers.glm.logit import LogisticRegression
from ml.utilities.metrics     import confusion_matrix

from sklearn import preprocessing, model_selection, metrics


def main(argv):
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)


    print()
    print('Classification Experiment: Diabetes')
    print()


    data  = np.loadtxt('./data/csv/diabetes.csv', delimiter = ',')
    X     = preprocessing.scale(data[:, :8])
    Y     = data[:, 8].astype(int)


    X, X_t, Y, Y_t = model_selection.train_test_split(X, Y, train_size = 0.75)



    logit = LogisticRegression()
    logit.fit(X, Y)
    P     = logit.predict(X_t)


    Y_hat = np.round(P)



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

