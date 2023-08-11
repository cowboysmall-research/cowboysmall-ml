import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import style

from typing import List
from numpy.typing import NDArray


def scatterplot(name: str, title: str, O_s: List[NDArray[np.uint8]], L_s: List[str], X_d: NDArray[np.uint8], Y_d: NDArray[np.uint8]) -> None:
    style.use("ggplot")

    plt.clf()
    plt.title(title)

    plt.legend([plt.scatter(o[:, 0], o[:, 1]) for o in O_s], L_s)
    plt.plot(X_d, Y_d)

    out_dir = './images/scatterplot/{}/'.format(name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig('{}/{}.png'.format(out_dir, name), format = 'png')
    plt.close()
