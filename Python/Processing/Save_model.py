import os
import re
import pandas as pd
import joblib


def save_Preceptron(estimator, input_set, target_set, path, overwrite=False):
    """
    save the estimator with trained parameter
    print save path and type of estimator
    :param estimator: [estimator],  MLP Perceptron model
    :param input_set: [narray],  Input data set, use to predict result
    :param target_set: [narray],  Target data set, use to determine estimator class
    :param path: [str],  saving path
    :param overwrite: [boolean],  whether overwrite existed model and prediction result
    """
    # determine estimator class
    try:
        target_set.shape[1]
    except IndexError:
        estimator_class = 'classifier'
    else:
        estimator_class = 'regressor'

    # determine path
    if not os.path.exists(path):
        os.makedirs(path)

    # assign depth of MLP
    hyperparameter = estimator.get_params()

    # make sure that 1 layer MLP could also have depth
    try:
        len(hyperparameter["hidden_layer_sizes"])
    except TypeError:
        deep = 1
    else:
        deep = len(hyperparameter["hidden_layer_sizes"])

    # determine joblib file and save
    model_name = f"{path}/{estimator_class}_layer_{deep}.joblib"
    if not os.path.exists(model_name):
        joblib.dump(estimator, model_name)
    elif overwrite:
        joblib.dump(estimator, model_name)

    # assign prediction result
    MLP_prediction = estimator.predict(input_set)
    MLP_prediction_df = pd.DataFrame.from_dict(MLP_prediction)

    # determine csv file and save
    prediction_name = f"{path}/{estimator_class}_layer_{deep}.csv"
    if not os.path.exists(prediction_name):
        MLP_prediction_df.to_csv(prediction_name)
    elif overwrite:
        MLP_prediction_df.to_csv(prediction_name)

    # assign target Dataframe
    MLP_target = pd.DataFrame.from_dict(target_set)

    # determine csv file and save
    target_name = f"{path}/{estimator_class}_target.csv"
    if not os.path.exists(target_name):
        MLP_target.to_csv(target_name)
    elif overwrite:
        MLP_target.to_csv(target_name)
        print("model and prediction result save at {} successfully".format(path))

    print("estimator class: {}".format(estimator_class))
    print("data version: {}".format(re.search(r'(?<=/)\w+', model_name).group(0)))