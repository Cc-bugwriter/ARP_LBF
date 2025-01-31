import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import zipfile
from Evaluation import plot_learning_curve as plc
from Evaluation import hyper_search as search
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from warnings import simplefilter


class Classification():
    """
    define all classification with enumeration
    """
    m2 = 1
    m3 = 10
    m4 = 100
    k = 1000
    a = 10000
    b = 100000
    belong_to = \
        {0: 'none',
          m2: 'm2', m3: 'm3', m4: 'm4', k: 'k', a: 'alpha', b: 'beta',
          m2 + m3: 'm23', m2 + m4: 'm24', m2 + k: 'm2k', m2 + a: 'm2a', m2 + b: 'm2b',
          m3 + m4: 'm34', m3 + k: 'm3k', m3 + a: 'm3a', m3 + b: 'm3b',
          m4 + k: 'm4k', m4 + a: 'm4a', m4 + b: 'm4b',
          k + a: 'ka', k + b: 'kb',
          a + b: 'ab',
          m2 + m3 + m4: 'm234', m2 + m3 + k: 'm23k', m2 + m3 + a: 'm23a', m2 + m3 + b: 'm23b', m2 + m4 + k: 'm24k',
          m2 + m4 + a: 'm24a', m2 + m4 + b: 'm24b', m2 + k + a: 'm2ka', m2 + k + b: 'm2kb', m2 + a + b: 'm2ab',
          m3 + m4 + k: 'm34k', m3 + m4 + a: 'm34a', m3 + m4 + b: 'm34b', m3 + k + a: 'm3ka', m3 + k + b: 'm3kb',
          m3 + a + b: 'm3ab',
          m4 + k + a: 'm4ka', m4 + k + b: 'm4kb', m4 + a + b: 'm4ab',
          k + a + b: 'kab',
          m2 + m3 + m4 + k: 'm234k', m2 + m3 + m4 + a: 'm234a', m2 + m3 + m4 + b: 'm234b', m2 + m3 + k + a: 'm23ka',
          m2 + m3 + k + b: 'm23kb', m2 + m3 + a + b: 'm23ab', m2 + m4 + k + a: 'm24ka', m2 + m4 + k + b: 'm24kb',
          m2 + m4 + a + b: 'm24ab', m2 + k + a + b: 'm2kab',
          m3 + m4 + k + a: 'm34ka', m3 + m4 + k + b: 'm34kb', m3 + m4 + a + b: 'm34ab', m3 + k + a + b: 'm3kab',
          m4 + k + a + b: 'm4kab',
          m2 + m3 + m4 + k + a: 'm234ka', m2 + m3 + m4 + k + b: 'm234kb', m2 + m3 + m4 + a + b: 'm234ab',
          m2 + m3 + k + a + b: 'm23kab', m2 + m4 + k + a + b: 'm24kab', m3 + m4 + k + a + b: 'm34kab',
          m2 + m3 + m4 + k + a + b: 'm234kab'}


def dataset_reader(path='Data/daten', name='1P1K', type='csv'):
    """
    read the data set
    and return Input and Target of Training Network
    :param path: [str], the path of csv data (default value : 'Data/daten')
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
    # regularization Input
    input_set = preprocessing.scale(input_set)

    # compute feature scaling parameter
    input_std = np.std(input_set, ddof=0)  # standard deviation with bias (column)
    input_mean = input_set.mean(0)  # mean (column)

    # assign binary mask to Target classification
    if target_set is not None:
        # assign reference value
        target_set_ref = [1, 1, 1, 2, 0.6261, 0.0001]
        # decide change of target parameter
        for i, reference in enumerate(target_set_ref):
            target_set[:, i] = target_set[:, i] != reference
        # convert boolen to int
        target_set = target_set.astype(int)

        # convert [5*1] array to str
        for i in range(target_set.shape[1]):
            target_set[:, i] = target_set[:, i] * 10 ** i
        target_set_list = np.sum(target_set, axis=1, dtype=int).tolist()

        target_name = []
        for i in target_set_list:
            target_name.append(Classification.belong_to[i])
        target_name = np.array(target_name)

        return input_set, input_std, input_mean, target_set, target_name

        return input_set, input_std, input_mean, target_set

    return input_set, input_std, input_mean


def regression(input_set, target_set, test_size=0.2, random_seed=23, alpha=1.3738e-4, hidden_layer_sizes=(46, 29, 26), max_iter=1000):
    """
    modeling a MLP Regressor with random split all data set.
    after training print out test score on console.

    :param input_set: [narray],  Input of Training Network
    :param target_set: [narray],  Target of Training Network
    :param test_size: [float], the proportion of test data in all data set (default value : 0.2)
    :param random_seed: [int], the random seed of random split for data set (default value : 233)
    :param alpha: [float], regularisation coefficient in MLP Regressor (default value : 8e-3)
    :param hidden_layer_sizes: [tuple of int], structural hyperparameter in MLP Regressor (default value : (46, 29, 26))
    :param max_iter: [int], maximal iteration epoch in MLP Regressor (default value : 1000)

    :return regressor: [estimator],  MLP Regressor with
    :return score: [float], determination coefficient of test data set
    :return weight_matrix: [narray], weight matrix of training data set
    """

    # split into training and test set
    X_train, X_test, y_train, y_test = \
        train_test_split(input_set, target_set, test_size=test_size, random_state=random_seed)

    # setup a MLP Regressor 3 layers
    regressor = MLPRegressor(solver='lbfgs', alpha=alpha,
                       hidden_layer_sizes=hidden_layer_sizes, random_state=1, max_iter=max_iter)

    # fit Regressor to the training data
    regressor.fit(X_train, y_train)

    # compute and Print R2 Metrics
    score = regressor.score(X_test, y_test)
    print('Test R2 Score: %f' % score)

    # compute Weight matrix
    weight_matrix = regressor.fit(X_train, y_train).coefs_

    return regressor, score, weight_matrix


def evaluation_learning_curve(estimator, input_set, target_set,
                              title="(46, 29, 26)", test_size=0.2, train_sizes=np.linspace(0.01, 1.0, 25)):
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
    plt.show()


def hyper_search(estimator, input_set, target_set, deep=3, random_mode=True):
    """
    evaluate determination coefficient of variable regularisation coefficient,
    find the best result and visualize evaluation process

    :param estimator: [estimator],  MLP Perceptron model
    :param input_set: [narray],  Input data set
    :param target_set: [narray],  Target data set
    :param deep: [int],  the number of layer width (default value: 3)
    :param random_mode: [boolen],  choose to random search
    """
    # find the lease common multiple, which base on the number of input's and target's feature
    try:
        target_set.shape[1]
    except IndexError:
        # classification
        width = input_set.shape[1] * 63 / math.gcd(input_set.shape[1], 63)
        width = int(width)

        # assign possible neuron number in domain
        candidate_neuron = range(63, width)
        model = 'classification'

    else:
        # regression
        width = input_set.shape[1] * target_set.shape[1] / math.gcd(input_set.shape[1], target_set.shape[1])
        width = int(width)

        # assign possible neuron number in domain
        candidate_neuron = range(target_set.shape[1], width)
        model = 'regression'

    # initialize the hidden_layer_sizes
    hidden_layer_sizes = []

    # assign possible hidden_layer_sizes
    if deep == 5:
        if model == 'classification':
            zf = zipfile.ZipFile('Data/hidden_layer_sizes_5_clf.zip')
            df = pd.read_csv(zf.open('hidden_layer_sizes_5_clf.csv'))
            hidden_layer_sizes = [list(row) for row in df.values]
        elif model == 'classification':
            df = pd.read_csv('Data/hidden_layer_sizes_5_mlg.csv')
            hidden_layer_sizes = [list(row) for row in df.values]
    if deep == 4:
        for layer_4 in candidate_neuron:
            for layer_3 in candidate_neuron:
                for layer_2 in candidate_neuron:
                    for layer_1 in candidate_neuron:
                        if layer_4 < layer_3 and layer_3 < layer_2 and layer_2 < layer_1:
                            hidden_layer_sizes.append((layer_1, layer_2, layer_3, layer_4))
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
    param_space = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': ['relu', 'logistic'],
        'solver': ['lbfgs'],
        'alpha': np.logspace(-5, -2, 30),
        'max_iter': np.logspace(3, 4, 10)}

    # run hyper parameter search
    if random_mode:
        hyper_search = RandomizedSearchCV(estimator, param_distributions=param_space)
    else:
        hyper_search = GridSearchCV(estimator, param_grid=param_space)

    hyper_search.fit(input_set, target_set)

    # reprot result of grid search
    search_result = hyper_search.cv_results_
    search.report_search(search_result)

    candidates = np.flatnonzero(search_result['rank_test_score'] == 1)

    return search_result[['params']][candidates]


def merge_data():
    """
    merge all data in a couple of data sets
    :return input_set: [narray],  Input data set
    :return target_set: [narray], Target data set
    """
    input_set, target_set = dataset_reader(name='1PmitT')
    for i in range(2, 8):
        name = f"{i}PmitT"
        input_append, target_append = dataset_reader(name=name)
        input_set = np.concatenate((input_set, input_append), axis=0)
        target_set = np.concatenate((target_set, target_append), axis=0)

    return input_set, target_set


if __name__ == '__main__':
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    # load data and preprocess
    input_set, target_set = merge_data()

    # preprocess for regression
    input_set, _, _ = dataset_preprocess(input_set)

    # model MLP Regressor
    MLP_regression, _, _ = regression(input_set, target_set)

    # # evaluation
    # evaluation_learning_curve(MLP_regression, input_set, target_set)  # learning curve
    # plot_regularization(MLP_regression, input_set, target_set)  # regularization coefficient
    #
    # # random search
    # hyper_search(MLP_regression, input_set, target_set)
