import sys
import warnings

import numpy  as np
import pandas as pd

from ml.classifiers.dt.tree import DecisionTree
from ml.classifiers.dt.cost import gini, entropy
from ml.utilities.metrics   import confusion_matrix

from sklearn import model_selection, metrics


def main(argv):
    np.random.seed(1337)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)


    print()
    print('Classification Experiment: Spambase')
    print()


    data  = pd.read_csv('./data/csv/spambase.csv')
    X     = data.iloc[:, :57]
    Y     = data.iloc[:, 57]


    X, X_t, Y, Y_t = model_selection.train_test_split(X, Y, train_size = 0.75)


    # dt    = DecisionTree(cost = entropy)
    dt    = DecisionTree(cost = gini)
    dt.fit(X, Y)
    P     = dt.predict(X_t)


    print(dt)



    print()
    print()
    print()
    print('                   Result: {:.2f}% Correct'.format(100 * (Y_t == P).sum() / float(len(Y_t))))
    print()
    print('    Classification Report:')
    print()
    print(metrics.classification_report(Y_t, P))
    print()
    print('         Confusion Matrix:')
    print()
    print(confusion_matrix(Y_t, P))
    print()



if __name__ == "__main__":
    main(sys.argv[1:])

