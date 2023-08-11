import os
import gzip

import numpy as np


def load_fashion_data(type):
    labels_path = os.path.join('./data/gz', '%s-labels-idx1-ubyte.gz' % type)
    with gzip.open(labels_path, 'rb') as lblpath:
        labels = np.frombuffer(lblpath.read(), dtype = np.uint8, offset = 8)

    images_path = os.path.join('./data/gz', '%s-images-idx3-ubyte.gz' % type)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype = np.uint8, offset = 16).reshape(len(labels), 784)

    return images, labels
