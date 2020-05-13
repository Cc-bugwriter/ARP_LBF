from sklearn.externals import joblib


def load_Preceptron(target_set, path, deep=3):
    """
    load the estimator from path
    print save path and type of estimator
    :param target_set: [narray],  Target data set, use to determine estimator class
    :param path: [str],  saving path
    :param deep: [int], depth of MLP, help to find joblib file
    :return estimator: [estimator],  MLP Perceptron model
    """
    # determine estimator class
    try:
        target_set.shape[1]
    except IndexError:
        estimator_class = 'classifier'
    else:
        estimator_class = 'regressor'

    # determine joblib file and save
    model_name = f"{path}/{estimator_class}_layer_{deep}.joblib"

    # Load from file
    estimator = joblib.load(model_name)

    return estimator
