import sys
import warnings

import numpy  as np
import pandas as pd

from ml.classifiers.dt.tree import DecisionTree
from ml.classifiers.dt.cost import gini, entropy

from sklearn import model_selection, metrics


def main(argv):
    # np.random.seed(1337)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)


    print()
    print('Classification Experiment: Red Wine')
    print()


    data  = pd.read_csv('./data/csv/wine_red.csv', sep = ';')
    X     = data.iloc[:, :11]
    Y     = data.iloc[:, 11]


    X, X_t, Y, Y_true = model_selection.train_test_split(X, Y, train_size = 0.5)


    dt    = DecisionTree(cost = entropy)
    # dt    = DecisionTree(cost = gini)
    dt.fit(X, Y)
    P     = dt.predict(X_t)


    print(dt)



    print()
    print()
    print()
    print('                   Result: {:.2f}% Correct'.format(100 * (Y_true == P).sum() / float(len(Y_true))))
    print()
    print('    Classification Report:')
    print()
    print(metrics.classification_report(Y_true, P))
    print()
    print('         Confusion Matrix:')
    print()
    print(metrics.confusion_matrix(Y_true, P))
    print()



if __name__ == "__main__":
    main(sys.argv[1:])

