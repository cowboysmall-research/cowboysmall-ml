import sys
import warnings

import numpy as np

from ml.classifiers.nn.network  import Network
from ml.classifiers.nn.layer    import InputLayer, HiddenLayer, OutputLayer
from ml.utilities.preprocessing import one_hot

from sklearn import preprocessing, model_selection, metrics


def main(argv):
    np.random.seed(1341)
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)


    print()
    print('Classification Experiment: Seeds')
    print()


    data = np.loadtxt('./data/csv/seeds.csv', delimiter = ',')
    X    = preprocessing.scale(data[:, :7])
    Y    = data[:, 7].astype(int)

    ohe  = one_hot.OneHotEncoder(Y)

    X, X_t, Y, Y_t = model_selection.train_test_split(X, ohe.encode(Y), train_size = 0.75)


    nn   = Network()

    nn.add(InputLayer(7,  learning = 0.5, regular = 0, momentum = 0))
    nn.add(HiddenLayer(7, learning = 0.5, regular = 0, momentum = 0))
    nn.add(HiddenLayer(7, learning = 0.5, regular = 0, momentum = 0))
    nn.add(OutputLayer(3))

    nn.fit(X, Y, batch = 100, epochs = 1000)

    P    = nn.predict(X_t)


    P    = ohe.decode(P)
    Y_t  = ohe.decode(Y_t)



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

