import os
from Processing import Load_model
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def regression(input_set, target_set, alpha=1.3738e-4, test_size=0.2,random_seed=233,
               hidden_layer_sizes=(46, 29, 26), max_iter=1000,
               hyperparameter=None, version="PmitT"):
    """
    modeling a MLP Regressor with random split all data set.
    after training print out test score on console.

    :param input_set: [narray],  Input of Training Network
    :param target_set: [narray],  Target of Training Network
    :param test_size: [float], the proportion of test data in all data set (default value : 0.2)
    :param random_seed: [int], the random seed of random split for data set (default value : 233)
    :param alpha: [float], regularisation coefficient in MLP Regressor (default value : 1.3738e-4,)
    :param hidden_layer_sizes: [tuple of int], structural hyperparameter in MLP Regressor (default value : (46, 29, 26))
    :param max_iter: [int], maximal iteration epoch in MLP Regressor (default value : 1000)
    :param hyperparameter: [dic], optimal hyper parameter, which comes from hyper search
    :param version: [str], version of data set, to assign the model path (default value: "PmitT")

    :return regressor: [estimator],  MLP Regressor with
    :return score: [float], determination coefficient of test data set
    :return weight_matrix: [narray], weight matrix of training data set
    """

    # split into training and test set
    X_train, X_test, y_train, y_test = \
        train_test_split(input_set, target_set, test_size=test_size, random_state=random_seed)

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
    print(model_path)

    # check whether a trained model exists
    if os.path.exists(model_path) and hyperparameter is None:
        # if exists a trained model, direct load
        regressor = Load_model.load_Preceptron(target_set, path=f"Model_parameters/{version}", deep=deep)
    else:
        # fit Regressor to the training data
        regressor.fit(X_train, y_train)

    # compute and Print R2 Metrics
    score = regressor.score(X_test, y_test)
    print('Test R2 Score: %f' % score)

    # predict training result
    y_pred = regressor.predict(X_test)

    # compute and Print other Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print('mean absolute error in Test: %f' % mae)
    print('mean squared error in Test: %f' % mse)

    return regressor