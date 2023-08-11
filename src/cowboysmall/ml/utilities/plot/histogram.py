import os
import datetime

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from matplotlib import style


def histogram(name, title, results, n_bins = 50, b_density = None, b_log = False, x_label = '', y_label = ''):
    style.use("ggplot")

    plt.clf()
    plt.title(title)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid(True)

    mean = np.mean(results)
    std  = np.std(results)

    x_loc = int(min(results))
    y_loc = int(max(results)) + 1

    plt.text(x_loc, y_loc, r'$\mu = {:8.2f},\ \sigma={:8.2f}$'.format(mean, std))

    _, bins, _ = plt.hist(results, bins = n_bins, density = b_density, log = b_log)
    plt.plot(bins, st.norm.pdf(bins, mean, std), linewidth = 2)

    out_dir = './images/{}/histogram'.format(datetime.date.today().strftime('%d%m%Y'))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig('{}/{}.png'.format(out_dir, name), format = 'png')
    plt.close()
