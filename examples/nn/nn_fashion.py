import os
import sys
import gzip

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

    X, Y     = load_fashion_data('train')
    X_t, Y_t = load_fashion_data('t10k')

    X   = preprocessing.scale(X)
    X_t = preprocessing.scale(X_t)

    ohe = OneHotEncoder()
    ohe.fit(np.concatenate((Y, Y_t)))

    nn = Network()
    nn.add(InputLayer(784,  learning = 0.1, regular = 0.01, momentum = 0.01))
    nn.add(HiddenLayer(256, learning = 0.1, regular = 0.01, momentum = 0, function = LeakyReLU()))
    nn.add(HiddenLayer(256, learning = 0.1, regular = 0.01, momentum = 0, function = LeakyReLU()))
    nn.add(OutputLayer(10))
    nn.fit(X, ohe.transform(Y), batch = 1200, epochs = 40)

    P = ohe.inverse_transform(nn.predict(X_t))

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


def load_fashion_data(type):
    labels_path = os.path.join('./data/gz', '%s-labels-idx1-ubyte.gz' % type)
    with gzip.open(labels_path, 'rb') as lblpath:
        labels = np.frombuffer(lblpath.read(), dtype = np.uint8, offset = 8)

    images_path = os.path.join('./data/gz', '%s-images-idx3-ubyte.gz' % type)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype = np.uint8, offset = 16).reshape(len(labels), 784)

    return images, labels


if __name__ == "__main__":
    main(sys.argv[1:])
