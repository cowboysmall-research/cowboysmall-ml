import sys
import warnings

import numpy             as np
import matplotlib.pyplot as plt

from ml.classifiers.glm.logit import LogisticRegression

from matplotlib import style

from sklearn import preprocessing, model_selection, metrics


def main(argv):
    style.use("ggplot")
    np.seterr(all = 'ignore')
    warnings.simplefilter(action = 'ignore', category = FutureWarning)


    print()
    print('Classification Experiment: Student')
    print()


    X     = np.loadtxt('./data/csv/student_X.csv', delimiter = ',')
    Y     = np.loadtxt('./data/csv/student_Y.csv', dtype = np.int)


    logit = LogisticRegression()
    logit.fit(X, Y)
    P     = logit.predict(np.array([[20, 80]]))


    Y_hat = np.round(P)


    print()
    print()
    print('            exam 1: 20%')
    print('            exam 2: 80%')
    print()
    print('           outcome: {}'.format(Y_hat[0]))
    print('           p value: {:.5f}'.format(P[0]))
    print()



    X_p   = [min(X[:, 0]) - 5, max(X[:, 0]) + 5]
    Y_p   = ((logit.theta[1] * X_p) + logit.theta[0]) / -logit.theta[2]


    a     = plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], marker = '+', label = 'admitted')
    n     = plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], marker = 'o', label = 'not admitted')
    plt.plot(X_p, Y_p, label = 'decision boundary')
    plt.legend((a, n), ('admitted', 'not admitted'))
    plt.show()



if __name__ == "__main__":
    main(sys.argv[1:])

