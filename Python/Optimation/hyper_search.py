import math
import numpy as np
import pandas as pd
import zipfile
import os
from Processing import Load_model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def hyper_search(estimator, input_set, target_set, deep=3, random_mode=True, version="PmitT"):
    """
    evaluate determination coefficient of variable regularisation coefficient,
    find the best result and visualize evaluation process

    :param estimator: [estimator],  MLP Perceptron model
    :param input_set: [narray],  Input data set
    :param target_set: [narray],  Target data set
    :param deep: [int],  the number of layer width (default value: 3)
    :param random_mode: [boolean],  choose to random search or grid search (default value: True)
    :param version: [str], version of data set, to assign the model path (default value: "PmitT")
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
        estimator_class = 'classifier'

    else:
        # regression
        width = input_set.shape[1] * target_set.shape[1] / math.gcd(input_set.shape[1], target_set.shape[1])
        width = int(width)

        # assign possible neuron number in domain
        candidate_neuron = range(target_set.shape[1], width)
        estimator_class = 'regressor'

    # initialize the hidden_layer_sizes
    hidden_layer_sizes = []

    # assign saved model path
    model_path = f"Model_parameters/{version}/{estimator_class}_layer_{deep}.joblib"

    if not os.path.exists(model_path):
        # assign possible hidden_layer_sizes
        if deep == 5:
            if estimator_class == 'classifier':
                zf = zipfile.ZipFile('Data/hidden_layer_sizes_5_clf.zip')
                df = pd.read_csv(zf.open('hidden_layer_sizes_5_clf.csv'))
                hidden_layer_sizes = [list(row) for row in df.values]
            elif estimator_class == 'regressor':
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


    else:
        # load existed model
        estimator = Load_model.load_Preceptron(target_set, path=f"Model_parameters/{version}", deep=deep)

        # assign reference layer parameters
        ref_layer = estimator.get_params()["hidden_layer_sizes"]

        if deep == 3:
            candidate_neuron_3 = range(ref_layer[1] - 10, ref_layer[1] + 10)
            candidate_neuron_2 = range(ref_layer[1] - 10, ref_layer[1] + 10)
            candidate_neuron_1 = range(ref_layer[0] - 10, ref_layer[0] + 10)
            for layer_3 in candidate_neuron_3:
                for layer_2 in candidate_neuron_2:
                    for layer_1 in candidate_neuron_1:
                        if layer_3 < layer_2 and layer_2 < layer_1:
                            hidden_layer_sizes.append((layer_1, layer_2, layer_3))
        elif deep == 2:
            candidate_neuron_2 = range(ref_layer[1] - 10, ref_layer[1] + 10)
            candidate_neuron_1 = range(ref_layer[0] - 10, ref_layer[0] + 10)
            for layer_2 in candidate_neuron_2:
                for layer_1 in candidate_neuron_1:
                    if layer_2 < layer_1:
                        hidden_layer_sizes.append((layer_1, layer_2))
        elif deep == 1:
            candidate_neuron = range(ref_layer-10, ref_layer+10)
            for layer_1 in candidate_neuron:
                hidden_layer_sizes.append((layer_1))

    # assign full grid over all hyper parameters
    param_space = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': ['relu'],
        'solver': ['lbfgs'],
        'alpha': np.logspace(-5, -2, 30),
        'max_iter': np.logspace(3, 4, 10)}

    print('hyper searching parameters')

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
    candidate = candidates[0]

    return search_result['params'][candidate]


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