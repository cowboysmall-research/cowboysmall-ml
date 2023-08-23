import numpy as np

from cowboysmall.ml.utilities.preprocessing import LabelEncoder


def kfold(classifier, data, k = 5, features = -1, shuffle = True):
    if shuffle:
        np.random.shuffle(data)

    le = LabelEncoder()
    le.fit(data[:, features])

    folds = np.array_split(data, k)
    scores = np.empty(k)

    for i in range(k):
        f = list(folds)
        del f[i]

        train = np.concatenate(f)
        test = folds[i]

        X = train[:, :features]
        Y = train[:, features].astype(int)

        X_t = test[:, :features]
        Y_t = test[:, features].astype(int)

        classifier.fit(X, le.transform(Y))
        Y_hat = le.inverse_transform(classifier.predict(X_t))

        scores[i] = 100 * (Y_t == Y_hat).sum() / float(len(Y_t))

    return scores
