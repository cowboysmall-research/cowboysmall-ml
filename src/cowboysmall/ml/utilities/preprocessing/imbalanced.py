import pandas as pd


def oversample(data, column):
    count   = data[column].value_counts()
    maximum = max(count)
    labels  = data[column].unique()
    samples = [data[data[column] == label].sample(frac = round(maximum / float(count[label])), replace = True) for label in labels]

    return pd.concat(samples).sample(frac = 1).reset_index(drop = True)
