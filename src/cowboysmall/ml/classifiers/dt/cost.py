import numpy as np


def gini(splits, classes):
    total = sum(split.shape[0] for split in splits)
    gini  = 0

    for split in splits:

        sigma = 0
        for c in classes:
            prop = split[split == c].shape[0] / float(split.shape[0])
            if prop != 0:
                sigma += prop * prop

        gini += (1.0 - sigma) * (split.shape[0] / float(total))

    return gini

def entropy(splits, classes):
    total   = sum(split.shape[0] for split in splits)
    entropy = 0

    for split in splits:

        sigma = 0
        for c in classes:
            prop = split[split == c].shape[0] / float(split.shape[0])
            if prop != 0:
                sigma += prop * np.log2(prop)

        entropy += -sigma * (split.shape[0] / float(total))

    return entropy
