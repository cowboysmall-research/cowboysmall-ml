import sys
import warnings

import numpy as np

from cowboysmall.ml.classifiers.nb.nb    import NaiveBayes
from cowboysmall.ml.utilities.validation import KFold


def main(argv):
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    data = np.loadtxt('./data/csv/spambase.csv', delimiter = ',')

    validation = KFold()
    scores = validation.validate(NaiveBayes(), data)

    print()
    print('Classification Experiment: Spambase - K-Fold Cross Validation')
    print()
    print()
    print()
    for i in range(len(scores)):
        print('                 Result {:>3} : {:8.2f}% Correct'.format(i + 1, scores[i]))
    print()
    print('                        Mean: {:8.2f}'.format(scores.mean()))
    print('          Standard Deviation: {:8.2f}'.format(scores.std()))
    print()
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
