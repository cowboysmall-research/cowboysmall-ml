
import os
import datetime

import numpy             as np
import matplotlib.pyplot as plt
import scipy.stats       as st


def histogram(name, title, results, n_bins = 50, b_density = None, b_log = False, x_label = '', y_label = ''):

    mean  = np.mean(results)
    std   = np.std(results)

    x_loc = int(min(results))
    y_loc = int(max(results)) + 1

    plt.clf()
    plt.title(title)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid(True)

    plt.text(x_loc, y_loc, r'$\mu = {:8.2f},\ \sigma={:8.2f}$'.format(mean, std))

    _, bins, _ = plt.hist(results, bins = n_bins, density = b_density, log = b_log)
    plt.plot(bins, st.norm.pdf(bins, mean, std), linewidth = 2)

    out_dir = './images/{}/histogram'.format(datetime.date.today().strftime('%d%m%Y'))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig('{}/{}.png'.format(out_dir, name), format = 'png')
    plt.close()

