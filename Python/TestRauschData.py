import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from Preprocessing import pre_processing, dataset_reader
from Processing import Regressor, Classifier, Save_model
from Evaluation import plot_learning_curve
from Evaluation import confusion_matrix
from Optimation import hyper_search
from sklearn.neural_network import MLPRegressor, MLPClassifier


def main(model_type, hyperparameter=None, data_version="version_6", data_loc=1):
    """
    main function of MLP
    :param model_type: [str], MLP perceptron model ("Classifier" or "Regressor")
    :param hyperparameter: [dict], result of hyper search
    :param data_version: [str], version of data set ('version_1', 'version_2', 'version_3', e.g.),
    (default value: "version_4")
    :param data_loc:  [int], data set index (default value: 1)
    """
    # assign parameter save and load path
    parameter_path = f"Model_parameters/{data_version}"
    print(f'data path: {parameter_path}')
    print(f'data name: daten{data_loc}_rauschen_sigma')
    data_name = '_rauschen_sigma'

    if model_type == "Regressor":
        # preprocess for MLP preceptron
        X_test, y_test = dataset_reader.dataset_reader(data_version=data_version,  name=f'{data_loc}{data_name}')
        X_test = pre_processing.data_scaling(X_test, version=data_version)

        # training MLP preceptron
        regressor = Regressor.regression(X_train=X_test, y_train=y_test, X_test=X_test, y_test=y_test,
                                         hyperparameter=hyperparameter, version=data_version)

        # save model and prediction result
        Save_model.save_Preceptron(regressor, X_test, y_test, path=parameter_path, overwrite=True)


if __name__ == '__main__':
    # parameter space (control)
    data_loc = 3
    data_version = "version_8"
    evaluation = False

    # define type of Model
    model_type = "Regressor"

    # define record data
    metric_file = os.path.join("Model_parameters", data_version, "evaluation.txt")

    # record test data
    with open(metric_file, 'w+') as file:
        name_space = {"1": 'sigma 0.2', "2": 'sigma 0.5', "3": 'sigma 1.0'}
        file.write("\t".join([f"test data from {name_space[str(data_loc)]}, dataversion: {data_version}"]) + "\n")

    main(model_type, data_version=data_version, data_loc=data_loc)
