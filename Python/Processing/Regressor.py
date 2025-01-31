import os
import numpy as np
import time
from Processing import Load_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error


def regression(X_train, y_train, X_test, y_test, alpha=4.175e-05, hidden_layer_sizes=(53, 26, 25), max_iter=4700,
               hyperparameter=None, version="version_4"):
    """
    modeling a MLP Regressor with random split all data set.
    after training print out test score on console.

    :param input_set: [narray],  Input of Training Network
    :param target_set: [narray],  Target of Training Network
    :param test_size: [float], the proportion of test data in all data set (default value : 0.2)
    :param random_seed: [int], the random seed of random split for data set (default value : 233)
    :param alpha: [float], regularisation coefficient in MLP Regressor (default value : 4.175e-05)
    :param hidden_layer_sizes: [tuple of int], structural hyperparameter in MLP Regressor (default value : (46, 29, 26))
    :param max_iter: [int], maximal iteration epoch in MLP Regressor (default value : 1000)
    :param hyperparameter: [dic], optimal hyper parameter, which comes from hyper search
    :param version: [str], version of data set, to assign the model path (default value: "version_4")

    :return regressor: [estimator],  MLP Regressor with
    :return score: [float], determination coefficient of test data set
    :return weight_matrix: [narray], weight matrix of training data set
    """

    # setup a MLP Regressor
    if hyperparameter is None:
        # MLP Regressor 3 layers (default)
        regressor = MLPRegressor(solver='lbfgs', alpha=alpha,
                                 hidden_layer_sizes=hidden_layer_sizes, random_state=1, max_iter=max_iter)
        # assign default depth
        deep = len(hidden_layer_sizes)
    else:
        # update hyper parameter base on hyper search
        regressor = MLPRegressor(solver='lbfgs', random_state=1)
        regressor.set_params(**hyperparameter)

        # assign depth
        # make sure that 1 layer MLP could also have depth
        try:
            len(hyperparameter["hidden_layer_sizes"])
        except TypeError:
            deep = 1
        else:
            deep = len(hyperparameter["hidden_layer_sizes"])

    # assign saved model path
    model_path = f"Model_parameters/{version}/regressor_layer_{deep}.joblib"
    print(f'model path: {model_path}')

    # check whether a trained model exists
    if os.path.exists(model_path) and hyperparameter is None:
        # if exists a trained model, direct load
        regressor = Load_model.load_Preceptron(y_train, path=f"Model_parameters/{version}", deep=deep)
    else:
        # timer start
        time_start = time.time()
        # fit Regressor to the training data
        regressor.fit(X_train, y_train)
        # timer end
        time_end = time.time()
        # print fitting time
        print('fitting time cost', time_end - time_start, 's')

    # compute and Print R2 Metrics
    score = regressor.score(X_test, y_test)
    print('Test R2 Score: %f' % score)

    # predict training result
    y_pred = regressor.predict(X_test)

    # assign name space
    name_space = ['m2', 'm3', 'm4', 'k', 'alpha', 'beta']

    # compute and Print other Metrics
    # save the metrics in txt data
    metric_file = os.path.join("Model_parameters", version, "evaluation.txt")
    print('depth of model: %d' % len(regressor.get_params()["hidden_layer_sizes"]))

    with open(metric_file, 'a+') as file:
        file.write("\t".join(['----------------------------']) + "\n")
        file.write("\t".join(['Test R2 Score: %f' % score]) + "\n")
        for i in range(y_test.shape[1]):
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            print('%s mean absolute error in Test: %f' % (name_space[i], mae))
            print('%s normal mean absolute error: %f in percent' % (name_space[i], 100 * mae / np.ptp(y_pred[:, i])))
            file.write("\t".join(['%s mean absolute error in Test: %f' % (name_space[i], mae)]) + "\n")
            file.write("\t".join(['%s normal mean absolute error: %f in percent' %
                       (name_space[i], 100 * mae / np.ptp(y_pred[:, i])) + "\n"]) + "\n")
        print("")

    return regressor