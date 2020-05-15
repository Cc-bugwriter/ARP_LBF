import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Preprocessing import pre_processing, dataset_reader
from Processing import Regressor, Classifier, Save_model
from Evaluation import plot_learning_curve
from Evaluation import confusion_matrix
from Optimation import hyper_search
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from warnings import simplefilter


def main(model_type, hyperparameter=None, data_version="version_4", evaluation=False):
    """
    main function of MLP
    :param model_type: [str],  MLP perceptron model ("Classifier" or "Regressor")
    :param hyperparameter: [dict], result of hyper search
    :param data_version: [str], version of data set ('version_1', 'version_2', 'version_3', e.g.),
    (default value: "version_4")
    :param evaluation: [boolean], determination, whether evaluate the fitting process or not
    """
    # load data set
    input_set, target_set = dataset_reader.merge_data(data_version)

    # assign parameter save and load path
    parameter_path= f"Model_parameters/{data_version}"

    if model_type == "Regressor":
        # preprocess for MLP preceptron
        input_set, _, _ = pre_processing.dataset_preprocess(input_set)

        # training MLP preceptron
        regressor = Regressor.regression(input_set, target_set, hyperparameter=hyperparameter,
                                                               version=data_version)

        # split into training and test set
        _, X_test, _, y_test = \
            train_test_split(input_set, target_set, test_size=0.2, random_state=23)

        # save model and prediction result
        Save_model.save_Preceptron(regressor, X_test, y_test, path=parameter_path)

        # evaluate fitting process
        if evaluation:
            plot_learning_curve.evaluation_learning_curve(regressor, X_test, y_test,
                                                          title=f"{hyperparameter['hidden_layer_sizes']}")

    elif model_type == "Classifier":
        # preprocess for MLP preceptron
        input_set, _, _, target_set, target_name = pre_processing.dataset_preprocess(input_set, target_set)

        # training MLP preceptron
        classifier = Classifier.classifier(input_set, target_set, hyperparameter=hyperparameter)

        # split into training and test set
        _, X_test, _, y_test = \
            train_test_split(input_set, target_set, test_size=0.2, random_state=23)

        # save model and prediction result
        Save_model.save_Preceptron(classifier, X_test, y_test, path=parameter_path)

        # evaluate MLP preceptron
        if evaluation:
            confusion_matrix.confusion_matrix(classifier, X_test, y_test, target_name)


def optimize(model, deep=3, data_version="version_4"):
    """
    optimize function of MLP
    :param model: [str],  MLP perceptron model ("Classifier" or "Regressor")
    :param deep: [int], depth of MLP Network  (default value: 3)
    :param data_version: [str], version of data set ('version_1', 'version_2', 'version_3', e.g.),
     (default value: "version_4")
    :return para_space: [dict], MLP Hyper parameter
    """
    # load data set
    input_set, target_set = dataset_reader.merge_data(data_version=data_version)

    if model == "Regressor":
        # preprocess for MLP preceptron
        input_set, _, _ = pre_processing.dataset_preprocess(input_set)

        # split into training and test set
        X_train, X_test, y_train, y_test = \
            train_test_split(input_set, target_set, test_size=0.2, random_state=233)

        # setup a MLP preceptron
        regressor = MLPRegressor(solver='lbfgs', random_state=1)

        # implement hyper search
        regressor_search = hyper_search.hyper_search(regressor, X_train, y_train, deep=deep, version=data_version)

        return regressor_search

    elif model == "Classifier":
        # preprocess for MLP preceptron
        input_set, _, _, target_set, target_name = pre_processing.dataset_preprocess(input_set, target_set)

        # split into training and test set
        X_train, X_test, y_train, y_test = \
            train_test_split(input_set, target_set, test_size=0.2, random_state=23)

        # setup a MLP preceptron
        classifier = MLPClassifier(solver='lbfgs', random_state=1)

        # implement hyper search
        classifier_search = hyper_search.hyper_search(classifier, X_train, y_train, deep=deep, version=data_version)

        return classifier_search


if __name__ == '__main__':

    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    # define type of Model
    model_type = "Regressor"
    # model_type = "Classifier"
    #
    # optimize hyper parameter
    deep_space = np.linspace(1, 3, num=3)
    for deep in deep_space:
        parameter_space = optimize(model_type, deep=int(deep))

        # train MLP
        main(model_type, parameter_space)

    # full path： "Model_parameters/version_4/classifier_layer_1.joblib"
    model_type = "Classifier"
    deep_space = np.linspace(1, 1, num=1)
    for deep in deep_space:
        parameter_space = optimize(model_type, deep=int(deep))

        # train MLP
        main(model_type, parameter_space)