import math
import numpy as np
import pandas as pd
import zipfile
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def hyper_search(estimator, input_set, target_set, deep=3, random_mode=True):
    """
    evaluate determination coefficient of variable regularisation coefficient,
    find the best result and visualize evaluation process

    :param estimator: [estimator],  MLP Perceptron model
    :param input_set: [narray],  Input data set
    :param target_set: [narray],  Target data set
    :param deep: [int],  the number of layer width (default value: 3)
    :param random_mode: [boolen],  choose to random search or grid search (default value: True)
    """
    # find the lease common multiple, which base on the number of input's and target's feature
    try:
        target_set.shape[1]
    except IndexError:
        # classification
        width = input_set.shape[1] * 63 / math.gcd(input_set.shape[1], 63)
        width = int(width)

        # assign possible neuron number in domain
        candidate_neuron = range(63, min(width, 2*63))
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
                            hidden_layer_sizes.append((layer_1, layer_2, layer_3,layer_4))
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
        'activation': ['relu'],
        'solver': ['lbfgs'],
        'alpha': np.logspace(-5, -2, 30),
        'max_iter': np.logspace(3, 4, 10)}

    # run hyper parameter search
    if random_mode:
        hyper_search = RandomizedSearchCV(estimator, param_distributions=param_space, n_jobs=-1)
    else:
        hyper_search = GridSearchCV(estimator, param_grid=param_space, n_jobs=-1)

    hyper_search.fit(input_set, target_set)

    # reprot result of grid search
    search_result = hyper_search.cv_results_
    report_search(search_result)

    candidates = np.flatnonzero(search_result['rank_test_score'] == 1)

    return search_result[['params']][candidates]


def report_search(results, n_top=3):
    """
    evaluate determination coefficient of variable regularisation coefficient,
    find the best result and visualize evaluation process

    :param results: [estimator],  MLP Perceptron model
    :param input_set: [narray],  Input data set

    Returns
    -------

    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")