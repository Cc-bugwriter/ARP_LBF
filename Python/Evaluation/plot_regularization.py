import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def plot_regularization(estimator, input_set, target_set, alphas=np.logspace(-5, -2, 30)):
    """
    evaluate determination coefficient of variable regularisation coefficient,
    find the best result and visualize evaluation process

    :param estimator: [estimator],  MLP Perceptron model
    :param input_set: [narray],  Input data set
    :param target_set: [narray],  Target data set
    :param alphas: [narray],  regularisation coefficient domain
    """
    # initialize figure
    plt.subplots(1, 1, figsize=(20, 20))

    # split into training and test set
    X_train, X_test, y_train, y_test = \
        train_test_split(input_set, target_set, test_size=0.2, random_state=233)

    # initialize list
    train_scores = list()
    test_scores = list()

    # traversal predefined regularisation coefficient domain
    for alpha in alphas:
        estimator.set_params(alpha=alpha)
        estimator.fit(X_train, y_train)
        train_scores.append(estimator.score(X_train, y_train))
        test_scores.append(estimator.score(X_test, y_test))

    # find the best alpha in domain
    i_alpha_optim = np.argmax(test_scores)
    alpha_optim = alphas[i_alpha_optim]
    print("Optimal regularization parameter : %s" % alpha_optim)

    # Estimate the loss function on full data with optimal regularization parameter
    estimator.set_params(alpha=alpha_optim)
    knn_loss = estimator.fit(input_set, target_set).loss_
    print("loss function at optimal alpha : %s" % knn_loss)

    plt.semilogx(alphas, train_scores, label='Train')
    plt.semilogx(alphas, test_scores, label='Test')
    plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_scores), color='k',
               linewidth=3, label='Optimum on test')
    plt.legend(loc='lower left')
    plt.ylim([0.95, 1.01])
    plt.xlabel('Regularization parameter')
    plt.ylabel('Performance')
    plt.legend()