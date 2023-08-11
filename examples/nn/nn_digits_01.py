import sys
import warnings

import numpy as np

from sklearn import preprocessing, model_selection, metrics

from cowboysmall.ml.classifiers.nn.network import Network
from cowboysmall.ml.classifiers.nn.layer   import InputLayer, HiddenLayer, OutputLayer
from cowboysmall.ml.utilities.function     import LeakyReLU
from cowboysmall.ml.utilities.metrics      import confusion_matrix


def main(argv):
    np.random.seed(1337)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    data = np.loadtxt('./data/csv/digits.csv', delimiter = ',')
    X = preprocessing.scale(data[:, 10:])
    Y = data[:, :10].astype(int)

    X, X_t, Y, Y_t = model_selection.train_test_split(X, Y, train_size = 0.75)

    nn = Network()
    nn.add(InputLayer(64,   learning = 0.25, regular = 0.001, momentum = 0.0125))
    nn.add(HiddenLayer(100, learning = 0.25, regular = 0.001, momentum = 0, function = LeakyReLU()))
    nn.add(HiddenLayer(100, learning = 0.25, regular = 0.001, momentum = 0, function = LeakyReLU()))
    nn.add(HiddenLayer(75,  learning = 0.25, regular = 0.001, momentum = 0.0125))
    nn.add(HiddenLayer(25,  learning = 0.25, regular = 0.001, momentum = 0.0125))
    nn.add(OutputLayer(10))
    nn.fit(X, Y, batch = 100, epochs = 1000)

    P   = np.array([p.argmax() for p in nn.predict(X_t)])
    Y_t = np.array([y.argmax() for y in Y_t])

    print()
    print()
    print('Classification Experiment: Digits')
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
