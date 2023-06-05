
import numpy as np


def confusion_matrix(y, y_hat):
    labels = sorted(list(set(y) | set(y_hat)))
    index  = dict((label, i) for i, label in enumerate(labels))

    cm     = np.zeros(shape = (len(labels), len(labels)))
    for (x, y) in zip(y, y_hat):
        cm[index[x]][index[y]] += 1

    return cm.astype(int)

