import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Preprocessing import pre_processing, dataset_reader
from Processing import Regressor, Classifier, Save_model
from Evaluation import plot_learning_curve
from Evaluation import confusion_matrix
from Optimation import hyper_search
from sklearn.neural_network import MLPRegressor, MLPClassifier
from warnings import simplefilter


def main(model_type, hyperparameter=None, data_version="PmitT", evaluation=False):
    """
    main function of MLP
    :param model_type: [str],  MLP perceptron model ("Classifier" or "Regressor")
    :param hyperparameter: [dict], result of hyper search
    :param data_version: [str], version of data set ('P', 'P1K', 'PmitT', e.g.), (default value: "PmitT")
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
        regressor, score, weight_matrix = Regressor.regression(input_set, target_set, hyperparameter=hyperparameter,
                                                               version=data_version)

        # save model and prediction result
        Save_model.save_Preceptron(regressor, input_set, target_set, path=parameter_path)

        # evaluate fitting process
        if evaluation:
            plot_learning_curve.evaluation_learning_curve(regressor, input_set, target_set,
                                                          title=f"{hyperparameter['hidden_layer_sizes']}")

    elif model_type == "Classifier":
        # preprocess for MLP preceptron
        input_set, _, _, target_set, target_name = pre_processing.dataset_preprocess(input_set, target_set)

        # training MLP preceptron
        classifier, score, weight_matrix = Classifier.classifier(input_set, target_set, hyperparameter=hyperparameter)

        # save model and prediction result
        Save_model.save_Preceptron(classifier, input_set, target_set, path=parameter_path)

        # evaluate MLP preceptron
        if evaluation:
            confusion_matrix.confusion_matrix(classifier, input_set, target_set, target_name)


def optimize(model, deep=3, data_version="PmitT"):
    """
    optimize function of MLP
    :param model: [str],  MLP perceptron model ("Classifier" or "Regressor")
    :param deep: [int], depth of MLP Network  (default value: 3)
    :param data_version: [str], version of data set ('P', 'P1K', 'PmitT', e.g.), (default value: "PmitT")
    :return para_space: [dict], MLP Hyper parameter
    """
    # load data set
    input_set, target_set = dataset_reader.merge_data()

    if model == "Regressor":
        # preprocess for MLP preceptron
        input_set, _, _ = pre_processing.dataset_preprocess(input_set)

        # setup a MLP preceptron
        regressor = MLPRegressor(solver='lbfgs', random_state=1)

        # implement hyper search
        regressor_search = hyper_search.hyper_search(regressor, input_set, target_set, deep=deep, version=data_version)

        return regressor_search

    elif model == "Classifier":
        # preprocess for MLP preceptron
        input_set, _, _, target_set, target_name = pre_processing.dataset_preprocess(input_set, target_set)

        # setup a MLP preceptron
        classifier = MLPClassifier(solver='lbfgs', random_state=1)

        # implement hyper search
        classifier_search = hyper_search.hyper_search(classifier, input_set, target_set, deep=deep, version=data_version)

        return classifier_search


if __name__ == '__main__':

    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    # define type of Model
    # model_type = "Regressor"
    model_type = "Classifier"

    # optimize hyper parameter
    deep_space = np.linspace(1, 1, num=1)
    for deep in deep_space:
        parameter_space = optimize(model_type, deep=deep)

        # train MLP
        main(model_type, parameter_space)
