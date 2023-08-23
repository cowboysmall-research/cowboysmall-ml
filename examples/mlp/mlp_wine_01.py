import sys
import warnings

import numpy  as np
import pandas as pd

from sklearn import preprocessing, model_selection, metrics

from cowboysmall.ml.classifiers.mlp.network import Network
from cowboysmall.ml.classifiers.mlp.layer   import InputLayer, HiddenLayer, OutputLayer
from cowboysmall.ml.utilities.preprocessing import OneHotEncoder, oversample
from cowboysmall.ml.utilities.metrics       import confusion_matrix


def main(argv):
    np.random.seed(2704)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    data = oversample(pd.read_csv('./data/csv/wine_red.csv', sep = ';'), 'quality')
    X = data.drop(['quality'], axis = 1).values
    Y = data.quality.values

    sclr = preprocessing.StandardScaler().fit(X)
    X    = sclr.transform(X)

    ohe = OneHotEncoder()
    ohe.fit(Y)

    X, X_t, Y, Y_t = model_selection.train_test_split(X, ohe.transform(Y), train_size = 0.5)

    nn = Network()
    nn.add(InputLayer(11,   learning = 0.1, regular = 0.005, momentum = 0.01))
    nn.add(HiddenLayer(100, learning = 0.1, regular = 0.005, momentum = 0))
    nn.add(HiddenLayer(100, learning = 0.1, regular = 0.005, momentum = 0))
    nn.add(OutputLayer(6))
    nn.fit(X, Y, batch = 400, epochs = 1000)

    P   = ohe.inverse_transform(nn.predict(X_t))
    Y_t = ohe.inverse_transform(Y_t)

    print()
    print()
    print('Classification Experiment: Red Wine')
    print()
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
