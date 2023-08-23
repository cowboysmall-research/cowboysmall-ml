import sys
import warnings

import numpy as np

from sklearn import preprocessing, model_selection, metrics

from cowboysmall.ml.classifiers.nn.network  import Network
from cowboysmall.ml.classifiers.nn.layer    import InputLayer, HiddenLayer, OutputLayer
from cowboysmall.ml.utilities.preprocessing import OneHotEncoder
from cowboysmall.ml.utilities.metrics       import confusion_matrix


def main(argv):
    np.random.seed(1337)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    data = np.loadtxt('./data/csv/iris_01.csv', delimiter = ',')
    X = preprocessing.scale(data[:, :4])
    Y = data[:, 4].astype(int)

    ohe = OneHotEncoder()
    ohe.fit(Y)

    X, X_t, Y, Y_t = model_selection.train_test_split(X, ohe.transform(Y), train_size = 0.75)

    nn = Network()
    nn.add(InputLayer(4,  learning = 0.1, regular = 0.01, momentum = 0.01, zero = False))
    nn.add(HiddenLayer(6, learning = 0.1, regular = 0.01, momentum = 0.01, zero = False))
    nn.add(HiddenLayer(6, learning = 0.1, regular = 0.01, momentum = 0.01, zero = False))
    nn.add(OutputLayer(3))
    nn.fit(X, Y, batch = 100, epochs = 1000)

    P   = ohe.inverse_transform(nn.predict(X_t))
    Y_t = ohe.inverse_transform(Y_t)

    print()
    print()
    print('Classification Experiment: Iris')
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
