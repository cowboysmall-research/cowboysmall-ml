import os
import datetime
import itertools

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from matplotlib import style


def confusion_matrix(name, title, y_true, y_hat, classes):
    style.use("ggplot")

    plt.clf()
    plt.title(title)

    plt.xlabel('Predicted')
    plt.ylabel('True')

    cm = metrics.confusion_matrix(y_true, y_hat)

    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment = 'center', color = 'white' if cm[i, j] > (cm.max() / 2.) else 'black')

    plt.tight_layout()

    out_dir = './images/{}/confusion_matrix'.format(datetime.date.today().strftime('%d%m%Y'))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig('{}/{}.png'.format(out_dir, name), format = 'png')
    plt.close()
