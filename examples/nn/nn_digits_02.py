import sys
import warnings

import numpy as np

from ml.classifiers.nn.network  import Network
from ml.classifiers.nn.layer    import InputLayer, HiddenLayer, OutputLayer
from ml.utilities.function import ReLU, LeakyReLU
from ml.utilities.preprocessing import one_hot
from ml.utilities.plot.confusion_matrix import confusion_matrix

from sklearn import preprocessing, metrics


def main(argv):
    np.random.seed(1337)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)


    print()
    print('Classification Experiment: Digits')
    print()


    train = np.loadtxt('./data/csv/digits_train.csv', delimiter = ',')
    X     = preprocessing.scale(train[:, :64])
    Y     = one_hot.forward(train[:, 64].astype(int))

    test  = np.loadtxt('./data/csv/digits_test.csv', delimiter = ',')
    X_t   = preprocessing.scale(test[:, :64])
    Y_t   = one_hot.forward(test[:, 64].astype(int))


    nn    = Network()

    # nn.add(InputLayer(64,  learning = 0.5, regular = 0.001, momentum = 0.0125))
    # nn.add(HiddenLayer(52, learning = 0.5, regular = 0.001, momentum = 0.0125))
    # nn.add(HiddenLayer(52, learning = 0.5, regular = 0.001, momentum = 0.0125))
    # nn.add(OutputLayer(10))

    nn.add(InputLayer(64,   learning = 0.25, regular = 0.001, momentum = 0.0125))
    nn.add(HiddenLayer(100, learning = 0.25, regular = 0.001, momentum = 0, function = LeakyReLU()))
    nn.add(HiddenLayer(100, learning = 0.25, regular = 0.001, momentum = 0, function = LeakyReLU()))
    nn.add(HiddenLayer(75,  learning = 0.25, regular = 0.001, momentum = 0.0125))
    nn.add(HiddenLayer(25,  learning = 0.25, regular = 0.001, momentum = 0.0125))
    nn.add(OutputLayer(10))

    nn.fit(X, Y, batch = 250, epochs = 500)
    # nn.fit(X, Y, batch = 250, epochs = 1000)

    P     = nn.predict(X_t)


    P     = one_hot.reverse(P)
    Y_t   = one_hot.reverse(Y_t)



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


    # confusion_matrix(Y_t, P, list(range(10)))



if __name__ == "__main__":
    main(sys.argv[1:])

