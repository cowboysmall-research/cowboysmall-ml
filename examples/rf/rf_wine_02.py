import sys
import warnings

import numpy  as np
import pandas as pd

from sklearn import model_selection, metrics

from cowboysmall.ml.classifiers.rf.random_forest import RandomForest
from cowboysmall.ml.classifiers.dt.cost          import gini, entropy
from cowboysmall.ml.utilities.preprocessing      import imbalanced
from cowboysmall.ml.utilities.metrics            import confusion_matrix


def main(argv):
    np.random.seed(1337)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)


    print()
    print('Classification Experiment: White Wine')
    print()


    data  = imbalanced.oversample(pd.read_csv('./data/csv/wine_white.csv', sep = ';'), 'quality')
    X     = data.iloc[:, :11]
    Y     = data.iloc[:, 11]


    X, X_t, Y, Y_true = model_selection.train_test_split(X, Y, train_size = 0.5)


    rf    = RandomForest(cost = entropy, s_ratio = 0.75, dt_count = 9, f_count = 6)
    # rf    = RandomForest(cost = gini, s_ratio = 0.75, dt_count = 9, f_count = 6)
    rf.fit(X, Y)
    P     = rf.predict(X_t)


    print(rf)



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
    print(confusion_matrix(Y_true, P))
    print()



if __name__ == "__main__":
    main(sys.argv[1:])

