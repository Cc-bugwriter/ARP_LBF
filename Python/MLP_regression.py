import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from time import time
from Evaluation import plot_learning_curve as plc
from Evaluation import report_search as report
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from warnings import simplefilter


def dataset_reader(path='Data/daten', name='1P1K', type='csv'):
    """
    read the data set
    and return Input and Target of Training Network
    :param path: [str], the path of csv data (default value : 'Data/rt-daten')
    :param name: [str], data to import ('1P' ,'1P1K' or '1PmitT', e.g.) (default value : '1P1K')
    :param type: [str], data type (default value : 'csv')
    :return input_set: [narray],  Input data set
    :return target_set: [narray], Target data set
    """
    # definition the data version
    datasets_v1 = ['1P', '2p', '3P', '4P', '5P', '6P', '7P']
    datasets_v2 = ['1P1K', '2P1K', '3P1K', '4P1K', '5P1K', '6P1K', '7P1K']
    datasets_v3 = ['1PmitT', '2PmitT', '3PmitT', '4PmitT', '5PmitT', '6PmitT', '7PmitT']

    # assign the data path, prepare for loading
    data = f"{path}{name}.{type}"
    # load the csv data with pandas
    df = pd.read_csv(data)

    # assign Input label
    label_x = ['omega_1', 'omega_2', 'omega_3', 'D_1', 'D_2', 'D_3', 'EVnorm1_1', 'EVnorm1_2', 'EVnorm1_3',
               'EVnorm2_1', 'EVnorm2_2', 'EVnorm2_3', 'EVnorm3_1', 'EVnorm3_2', 'EVnorm3_3']

    # switch Target label based on data version
    if name in datasets_v1:
        label_y = ['m2', 'm3', 'm4', 'k5', 'k6', 'alpha', 'beta']
    elif name in datasets_v2:
        label_y = ['m2', 'm3', 'm4', 'k5plusk6', 'alpha', 'beta']
    elif name in datasets_v3:
        label_y = ['m2', 'm3', 'm4', 'k', 'alpha', 'beta']
        label_x.append('Tem')


    input_set = df[label_x].values
    target_set = df[label_y].values

    return input_set, target_set


def dataset_preprocess(input_set, target_set=None):
    """
    regularise Input data set

    :param input_set: [narray],  Input data set
    :param target_set: [narray],  Target data set (default value: None)
    :return input_set: [narray],  Input data set after regularization
    :return input_std: [narray],  regularization standard deviation
    :return input_mean: [narray],  regularization mean
    """
    # Regularization Input
    input_set = preprocessing.scale(input_set)

    # Compute feature scaling parameter
    input_std = np.std(input_set, ddof=0)  # standard deviation with bias (column)
    input_mean = input_set.mean(0)  # mean (column)

    # assign binary mask to Target classification
    if target_set is not None:
        target_set = target_set != 1
        return input_set, input_std, input_mean, target_set

    return input_set, input_std, input_mean


def regression(input_set, target_set, test_size=0.2, random_seed=23, alpha=8e-3, hidden_layer_sizes=(105, 70, 46), max_iter=500):
    """
    modeling a MLP Regressor with random split all data set.
    after training print out test score on console.

    :param input_set: [narray],  Input of Training Network
    :param target_set: [narray],  Target of Training Network
    :param test_size: [float], the proportion of test data in all data set (default value : 0.2)
    :param random_seed: [int], the random seed of random split for data set (default value : 233)
    :param alpha: [float], regularisation coefficient in MLP Regressor (default value : 8e-3)
    :param hidden_layer_sizes: [tuple of int], structural hyperparameter in MLP Regressor (default value : (105, 70, 46))
    :param max_iter: [int], maximal iteration epoch in MLP Regressor (default value : (105, 70, 46))

    :return regressor: [estimator],  MLP Regressor with
    :return score: [float], determination coefficient of test data set
    :return weight_matrix: [narray], weight matrix of training data set
    """

    # Split into training and test set
    X_train, X_test, y_train, y_test = \
        train_test_split(input_set, target_set, test_size=test_size, random_state=random_seed)

    # Setup a MLP Regressor 3 layers
    regressor = MLPRegressor(solver='lbfgs', alpha=alpha,
                       hidden_layer_sizes=hidden_layer_sizes, random_state=1, max_iter=max_iter)

    # Fit Regressor to the training data
    regressor.fit(X_train, y_train)

    # Compute and Print R2 Metrics
    score = regressor.score(X_test, y_test)
    print('Test R2 Score: %f' % score)

    # Compute aWeight matrix
    weight_matrix = regressor.fit(X_train, y_train).coefs_

    return regressor, score, weight_matrix


def evaluation_learning_curve(estimator, input_set, target_set,
                              title="(105, 70, 46)", test_size=0.2, train_sizes=np.linspace(0.01, 1.0, 25)):
    """
    evaluate estimator fitting quality and performance, which implements on random cross validation.
    the data size of cross validation will linear increase, so that could research the overfitting and underfitting problem
    every cross validation is isolated, each data set dosen't have any influence of any other data sets.
    after each cross validation cache the score and time cost, finally will be visualized.

    :param estimator: [estimator],  MLP Perceptron model
    :param input_set: [narray],  Input data set
    :param target_set: [narray],  Target data set
    :param title: [str], string of structural hyperparameter in MLP Perceptron (default value : "(105, 70, 46)")
    :param test_size: [float], the proportion of test data in all data set (default value : 0.2)
    :param train_sizes: [nrarray], the proportion of cross validation data in all data set (default value : np.linspace(0.01, 1.0, 25))
    """
    # set the Title of Learning curve
    title = "Learning Curves" + title

    # randomly select cross validation set.
    cv = ShuffleSplit(test_size=test_size)

    # recall function
    plc.plot_learning_curve(estimator, title, input_set, target_set,
                            cv=cv, ylim=(0., 1.01), n_jobs=6, train_sizes=train_sizes)
    plt.show()


def plot_regularization(estimator, input_set, target_set):
    """
    evaluate determination coefficient of variable regularisation coefficient,
    find the best result and visualize evaluation process

    :param estimator: [estimator],  MLP Perceptron model
    :param input_set: [narray],  Input data set
    :param target_set: [narray],  Target data set
    """
    # initialize figure
    plt.subplots(1, 1, figsize=(20, 20))

    # split into training and test set
    X_train, X_test, y_train, y_test = \
        train_test_split(input_set, target_set, test_size=0.2, random_state=233)

    # assign regularisation coefficient domain
    alphas = np.logspace(-5, -2, 30)
    train_scores = list()  # initialize list
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
    plt.show()


def grid_search(estimator, input_set, target_set, deep=3):
    """
    evaluate determination coefficient of variable regularisation coefficient,
    find the best result and visualize evaluation process

    :param estimator: [estimator],  MLP Perceptron model
    :param input_set: [narray],  Input data set
    :param target_set: [narray],  Target data set
    :param deep: [int],  the number of layer width (default value: 3)
    """
    # find the lease common multiple, which base on the number of input's and target's feature
    width = input_set.shape[1]*target_set.shape[1]/math.gcd(input_set.shape[1], target_set.shape[1])
    width = int(width)

    # assign possible neuron number in domain
    candidate_neuron = range(target_set.shape[1], width)

    # initialize the hidden_layer_sizes
    hidden_layer_sizes = []

    # assign possible hidden_layer_sizes
    if deep == 3:
        for layer_3 in candidate_neuron:
            for layer_2 in candidate_neuron:
                for layer_1 in candidate_neuron:
                    if layer_3 < layer_2 and layer_2 < layer_1:
                        hidden_layer_sizes.append((layer_1, layer_2, layer_3))
    elif deep == 2:
        for layer_2 in candidate_neuron:
            for layer_1 in candidate_neuron:
                if layer_2 < layer_1:
                    hidden_layer_sizes.append((layer_1, layer_2))
    elif deep == 1:
        for layer_1 in candidate_neuron:
            hidden_layer_sizes.append((layer_1))

    # assign full grid over all hyper parameters
    param_grid = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': ['relu'],
        'solver': ['lbfgs'],
        'alpha': np.logspace(-5, -2, 30)}

    # run grid search
    grid_search = GridSearchCV(estimator, param_grid=param_grid)
    start = time()
    grid_search.fit(input_set, target_set)

    # reprot result of grid search
    report(grid_search.cv_results_)


if __name__ == '__main__':
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    # load data and preprocess
    input_set, target_set = dataset_reader(name='1PmitT')
    for i in range(2, 8):
        name = f"{i}PmitT"
        input_append, target_append = dataset_reader(name=name)
        input_set = np.concatenate((input_set, input_append), axis=0)
        target_set = np.concatenate((target_set, target_append), axis=0)

    input_set, _, _ = dataset_preprocess(input_set)  # preprocess for regression

    # input_set, _, _, target_set = dataset_preprocess(input_set, target_set)  # preprocess for classification

    # model MLP Regressor
    MLP_regression, _, _ = regression(input_set, target_set)

    # evaluation
    evaluation_learning_curve(MLP_regression, input_set, target_set)  # learning curve
    plot_regularization(MLP_regression, input_set, target_set)  # regularization coefficient

    # grid search
    grid_search(MLP_regression, input_set, target_set)