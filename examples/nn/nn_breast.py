import sys

import numpy as np

from sklearn import preprocessing, model_selection, metrics

from cowboysmall.ml.classifiers.nn.network  import Network
from cowboysmall.ml.classifiers.nn.layer    import InputLayer, HiddenLayer, OutputLayer
from cowboysmall.ml.utilities.function      import LeakyReLU
from cowboysmall.ml.utilities.preprocessing import OneHotEncoder
from cowboysmall.ml.utilities.metrics       import confusion_matrix


def main(argv):
    np.random.seed(1337)

    data = np.loadtxt('./data/csv/breast.csv', delimiter = ',')
    X = preprocessing.scale(data[:, 2:])
    Y = data[:, 1].astype(int)

    ohe = OneHotEncoder()
    ohe.fit(Y)

    X, X_t, Y, Y_t = model_selection.train_test_split(X, ohe.transform(Y), train_size = 0.75)

    nn = Network()
    nn.add(InputLayer(9,   learning = 0.1, regular = 0.001, momentum = 0.01))
    nn.add(HiddenLayer(18, learning = 0.1, regular = 0.001, momentum = 0.01, function = LeakyReLU()))
    nn.add(HiddenLayer(18, learning = 0.1, regular = 0.001, momentum = 0.01, function = LeakyReLU()))
    nn.add(OutputLayer(6))
    nn.fit(X, Y, batch = 50, epochs = 1000)

    P   = ohe.inverse_transform(nn.predict(X_t))
    Y_t = ohe.inverse_transform(Y_t)

    print()
    print()
    print('Classification Experiment: Breast')
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
