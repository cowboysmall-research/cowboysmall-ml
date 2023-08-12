import sys
import warnings

import numpy as np

from sklearn import preprocessing, model_selection, metrics

from cowboysmall.ml.classifiers.nn.network  import Network
from cowboysmall.ml.classifiers.nn.layer    import InputLayer, HiddenLayer, OutputLayer
from cowboysmall.ml.utilities.function      import LeakyReLU, ReLU
from cowboysmall.ml.utilities.preprocessing import OneHotEncoder
from cowboysmall.ml.utilities.metrics       import confusion_matrix

from examples.nn import load_fashion_data


def main(argv):
    np.random.seed(1337)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    X, Y     = load_fashion_data('train')
    X_t, Y_t = load_fashion_data('t10k')

    X   = preprocessing.scale(X)
    X_t = preprocessing.scale(X_t)

    ohe = OneHotEncoder(Y)

    nn = Network()
    nn.add(InputLayer(784,  learning = 0.1, regular = 0.01, momentum = 0.01))
    nn.add(HiddenLayer(256, learning = 0.1, regular = 0.01, momentum = 0))
    nn.add(HiddenLayer(256, learning = 0.1, regular = 0.01, momentum = 0))
    nn.add(OutputLayer(10))
    nn.fit(X, ohe.encode(Y), batch = 1000, epochs = 40)

    P = ohe.decode(nn.predict(X_t))

    print()
    print()
    print('Classification Experiment: Fashion')
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
