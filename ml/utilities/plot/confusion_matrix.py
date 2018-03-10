
import itertools

import numpy             as np
import matplotlib.pyplot as plt

from sklearn import metrics


def confusion_matrix(y_true, y_hat, classes):

    cm = metrics.confusion_matrix(y_true, y_hat)

    plt.figure()
    plt.title('Confusion Matrix')

    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment = 'center', color = 'white' if cm[i, j] > (cm.max() / 2.) else 'black')

    plt.tight_layout()

    plt.show()

