
import itertools

import numpy             as np
import matplotlib.pyplot as plt

from sklearn import metrics, preprocessing


def roc_curve(y_true, y_score, classes):

    plt.figure()
    plt.title('ROC Curve')

    plt.xlabel('False Positive')
    plt.ylabel('True Positive')

    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')

    y = preprocessing.label_binarize(y_true, classes)

    for i in range(len(classes)):
        fp, tp, _ = metrics.roc_curve(y[i], y_score[i])
        plt.plot(fp, tp, label = 'label {}'.format(classes[i]))

    plt.legend(loc = 'best')

    # plt.tight_layout()

    plt.show()

