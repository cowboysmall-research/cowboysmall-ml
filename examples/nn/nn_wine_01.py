import sys
import warnings

import numpy as np
import pandas as pd

from ml.classifiers.nn.network  import Network
from ml.classifiers.nn.layer    import InputLayer, HiddenLayer, OutputLayer 
from ml.utilities.preprocessing import one_hot

from sklearn import preprocessing, model_selection, metrics, feature_selection


def main(argv):
    np.random.seed(1337)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)


    print()
    print('Classification Experiment: Red Wine')
    print()


    data = pd.read_csv('./data/csv/wine_red.csv', sep = ';')
    X    = data.drop(['quality'], axis = 1).values
    Y    = data.quality.values - 3


    X    = feature_selection.SelectKBest(feature_selection.chi2, k = 9).fit_transform(X, Y)
    sclr = preprocessing.StandardScaler().fit(X)

    X, X_t, Y, Y_t = model_selection.train_test_split(X, one_hot.forward(Y), train_size = 0.75)

    X    = sclr.transform(X)
    X_t  = sclr.transform(X_t)


    nn   = Network()

    nn.add(InputLayer(9, learning = 0.5, regular = 0.01, momentum = 0.01))
    nn.add(HiddenLayer(12, learning = 0.5, regular = 0.01, momentum = 0.01))
    nn.add(HiddenLayer(12, learning = 0.5, regular = 0.01, momentum = 0.01))
    nn.add(HiddenLayer(12, learning = 0.5, regular = 0.01, momentum = 0.01))
    nn.add(HiddenLayer(12, learning = 0.5, regular = 0.01, momentum = 0.01))
    nn.add(OutputLayer(6))

    nn.fit(X, Y, batch = 200, epochs = 1000)

    P    = nn.predict(X_t)


    P    = one_hot.reverse(P)
    Y_t  = one_hot.reverse(Y_t)


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
    print(metrics.confusion_matrix(Y_t, P))
    print()



if __name__ == "__main__":
    main(sys.argv[1:])

