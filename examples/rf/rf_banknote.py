import sys

import numpy  as np
import pandas as pd

from sklearn import model_selection, metrics

from cowboysmall.ml.classifiers.rf.forest import RandomForest
from cowboysmall.ml.classifiers.dt.cost   import gini
from cowboysmall.ml.utilities.metrics     import confusion_matrix


def main(argv):
    np.random.seed(1337)

    data = pd.read_csv('./data/csv/banknote.csv', names = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class'])
    X = data.iloc[:, :4]
    Y = data.iloc[:, 4]

    X, X_t, Y, Y_t = model_selection.train_test_split(X, Y, train_size = 0.75)

    # rf = RandomForest(cost = entropy, s_ratio = 0.75, dt_count = 9, f_count = 3)
    rf = RandomForest(cost = gini, s_ratio = 0.75, dt_count = 9, f_count = 3)
    rf.fit(X, Y)
    P = rf.predict(X_t)

    print()
    print('Classification Experiment: Banknote')
    print()
    print(rf)
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
