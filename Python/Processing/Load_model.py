from sklearn.externals import joblib


def save_Preceptron(input_set, target_set, path, deep=3):
    """
    load the estimator from path
    print save path and type of estimator
    :param input_set: [narray],  Input data set, use to predict result
    :param target_set: [narray],  Target data set, use to determine estimator class
    :param path: [str],  saving path
    :return estimator: [estimator],  MLP Perceptron model
    """
    # determine estimator class
    try:
        target_set.shape[1]
    except IndexError:
        estimator_class = 'claasifier'
    else:
        estimator_class = 'regressor'

    # determine pkl file and save
    model_name = f"{path}/{estimator_class}_layer_{deep}.pkl"

    # Load from file
    estimator = joblib.load(model_name)

    return estimator
