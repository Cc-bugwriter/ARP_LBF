import os
import pandas as pd
from sklearn.externals import joblib


def save_Preceptron(estimator, input_set, target_set, path, overwrite=False):
    """
    save the estimator with trained parameter
    print save path and type of estimator
    :param estimator: [estimator],  MLP Perceptron model
    :param input_set: [narray],  Input data set, use to predict result
    :param target_set: [narray],  Target data set, use to determine estimator class
    :param path: [str],  saving path
    :param name: [str],  model name
    :param overwrite: [boolen],  whether overwrite existed model and prediction result
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
    deep = len(hyperparameter["hidden_layer_sizes"])

    # determine pkl file and save
    model_name = f"{path}/{estimator_class}_layer_{deep}.pkl"
    if not os.path.exists(model_name):
        joblib.dump(estimator, model_name)
    elif overwrite:
        joblib.dump(estimator, model_name)

    # assign prediction result
    MLP_prediction = estimator.predict(input_set)
    MLP_prediction_df = pd.DataFrame.from_dict(MLP_prediction)

    # determine csv file and save
    prediction_name = f"{path}/{estimator_class}_layer_{deep}.csv"
    if not os.path.exists(model_name):
        MLP_prediction_df.to_csv(prediction_name)
    elif overwrite:
        MLP_prediction_df.to_csv(prediction_name)

    print("estimator class: {}".format(estimator_class))
    print("model and prediction result saved at {}".format(path))