import sys
import warnings

import numpy as np

from sklearn import preprocessing, metrics

from cowboysmall.ml.classifiers.nn.network  import Network
from cowboysmall.ml.classifiers.nn.layer    import InputLayer, HiddenLayer, OutputLayer
from cowboysmall.ml.utilities.function      import LeakyReLU
from cowboysmall.ml.utilities.preprocessing import OneHotEncoder
from cowboysmall.ml.utilities.metrics       import confusion_matrix


def main(argv):
    np.random.seed(1337)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)


    print()
    print('Classification Experiment: Digits')
    print()


    train = np.loadtxt('./data/csv/digits_train.csv', delimiter = ',')
    X     = preprocessing.scale(train[:, :64])
    Y     = train[:, 64].astype(int)

    test  = np.loadtxt('./data/csv/digits_test.csv', delimiter = ',')
    X_t   = preprocessing.scale(test[:, :64])
    Y_t   = test[:, 64].astype(int)


    ohe   = OneHotEncoder(np.concatenate((Y, Y_t), axis = 0))


    nn    = Network()

    nn.add(InputLayer(64,   learning = 0.25, regular = 0.001, momentum = 0.0125))
    nn.add(HiddenLayer(100, learning = 0.25, regular = 0.001, momentum = 0, function = LeakyReLU()))
    nn.add(HiddenLayer(100, learning = 0.25, regular = 0.001, momentum = 0, function = LeakyReLU()))
    nn.add(HiddenLayer(75,  learning = 0.25, regular = 0.001, momentum = 0.0125))
    nn.add(HiddenLayer(25,  learning = 0.25, regular = 0.001, momentum = 0.0125))
    nn.add(OutputLayer(10))

    nn.fit(X, ohe.encode(Y), batch = 250, epochs = 500)

    P     = nn.predict(X_t)

    P     = ohe.decode(P)


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

