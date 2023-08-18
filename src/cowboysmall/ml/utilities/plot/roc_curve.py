import os
import datetime

import matplotlib.pyplot as plt

from sklearn import metrics, preprocessing
from matplotlib import style


def roc_curve(name, title, y_true, y_score, classes):
    style.use("ggplot")

    plt.clf()
    plt.title(title)

    plt.xlabel('False Positive')
    plt.ylabel('True Positive')

    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')

    y = preprocessing.label_binarize(y_true, classes)

    for i in range(len(classes)):
        fp, tp, _ = metrics.roc_curve(y[:, i], y_score[:, i])
        auc       = metrics.auc(fp, tp)
        plt.plot(fp, tp, label = 'label {:2d}: AUC {:0.2f}'.format(int(classes[i]), auc))

    plt.legend(loc = 'best')

    out_dir = './images/{}/roc_curve'.format(datetime.date.today().strftime('%d%m%Y'))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig('{}/{}.png'.format(out_dir, name), format = 'png')
    plt.close()
