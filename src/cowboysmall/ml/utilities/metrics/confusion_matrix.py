import numpy as np


def confusion_matrix(y, y_hat):
    labels = np.unique(np.concatenate((y, y_hat)))

    cm = np.zeros(shape = (labels.shape[0], labels.shape[0]))
    for (x, y) in zip(y, y_hat):
        cm[np.searchsorted(labels, x)][np.searchsorted(labels, y)] += 1

    return cm.astype(int)
