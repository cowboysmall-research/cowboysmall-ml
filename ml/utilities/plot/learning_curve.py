
import numpy             as np
import matplotlib.pyplot as plt

from sklearn import model_selection


def learning_curve(classifier, title, X, y, ylim = None, cv = None, jobs = 4, train_sizes = np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores = model_selection.learning_curve(classifier, X, y, cv = cv, n_jobs = jobs, train_sizes = train_sizes)

    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std  = np.std(train_scores,  axis = 1)
    test_scores_mean  = np.mean(test_scores,  axis = 1)
    test_scores_std   = np.std(test_scores,   axis = 1)

    plt.figure()
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training Data')
    plt.ylabel('Score')

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.1, color = 'r')
    plt.fill_between(train_sizes,  test_scores_mean - test_scores_std,   test_scores_mean + test_scores_std,  alpha = 0.1, color = 'g')

    plt.plot(train_sizes, train_scores_mean, 'o-', color = 'r', label = 'Training score')
    plt.plot(train_sizes, test_scores_mean,  'o-', color = 'g', label = 'Cross-validation score')

    plt.legend(loc = 'best')

    plt.show()

