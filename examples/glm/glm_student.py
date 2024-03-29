import sys

import numpy as np

from cowboysmall.ml.classifiers.glm.logit import LogisticRegression
from cowboysmall.ml.utilities.plot        import scatterplot


def main(argv):
    np.random.seed(1337)

    X = np.loadtxt('./data/csv/student_X.csv', delimiter = ',')
    Y = np.loadtxt('./data/csv/student_Y.csv', dtype = int)

    logit = LogisticRegression()
    logit.fit(X, Y)
    P = logit.predict(np.array([[20, 80]]))

    Y_hat = np.round(P)

    print()
    print('Classification Experiment: Student')
    print()
    print()
    print()
    print('            exam 1: 20%')
    print('            exam 2: 80%')
    print()
    print('           outcome: {}'.format(Y_hat[0]))
    print('           p value: {:.5f}'.format(P[0]))
    print()

    X_d = [min(X[:, 0]) - 5, max(X[:, 0]) + 5]
    Y_d = ((logit.theta[1] * X_d) + logit.theta[0]) / -logit.theta[2]
    scatterplot('glm_student', 'Student', [X[Y == 1], X[Y == 0]], ['admitted', 'not admitted'], X_d, Y_d)


if __name__ == "__main__":
    main(sys.argv[1:])
