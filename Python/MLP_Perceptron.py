import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Preprocessing import pre_processing, dataset_reader
from Processing import Regressor, Classifier, Save_model
from Evaluation import plot_learning_curve
from Evaluation import confusion_matrix
from Evaluation import hyper_search
from sklearn.neural_network import MLPRegressor, MLPClassifier
from warnings import simplefilter


def main(model, hyperparameter=None):
    """
    main function of NN
    :param model: [str],  MLP perceptron model ("Classifier" or "Regressor")
    """
    # load data set
    input_set, target_set = dataset_reader.merge_data()

    # preprocess for MLP preceptron
    if model == "Regressor":
        input_set, _, _ = pre_processing.dataset_preprocess(input_set)
    elif model == "Classifier":
        input_set, _, _, target_set, target_name = pre_processing.dataset_preprocess(input_set, target_set)

    # training MLP preceptron
    if model == "Regressor":
        regressor, score, weight_matrix = Regressor.regression(input_set, target_set, hyperparameter)
    elif model == "Classifier":
        classifier, score, weight_matrix = Classifier.classifier(input_set, target_set, hyperparameter)

    # evaluate MLP preceptron
    if model == "Regressor":
        plot_learning_curve.evaluation_learning_curve(regressor, input_set, target_set)
    elif model == "Classifier":
        confusion_matrix.confusion_matrix(classifier, input_set, target_set, target_name)


def optimize(model, deep=3, Data_version="PmitT"):
    """
    optimize function of MLP
    :param model: [str],  MLP perceptron model ("Classifier" or "Regressor")
    :param deep: [int], depth of MLP Network  (default value: 3)
    :return para_space: [dict], MLP Hyper parameter
    """
    # load data set
    input_set, target_set = dataset_reader.merge_data()

    # preprocess for MLP preceptron
    if model == "Regressor":
        input_set, _, _ = pre_processing.dataset_preprocess(input_set)
    elif model == "Classifier":
        input_set, _, _, target_set, target_name = pre_processing.dataset_preprocess(input_set, target_set)

    # setup a MLP preceptron
    if model == "Regressor":
        regressor = MLPRegressor(solver='lbfgs', random_state=1)
    elif model == "Classifier":
        classifier = MLPClassifier(solver='lbfgs', random_state=1)

    # search best hyper parameter
    if model == "Regressor":
        # implement hyper search
        regressor_search = hyper_search.hyper_search(regressor, input_set, target_set, deep=deep)

        # save model and prediction result
        Parameter_path = f"Model_parameters/{Data_version}"
        Save_model(regressor, input_set, target_set, path=Parameter_path)

        return regressor_search

    elif model == "Classifier":
        # implement hyper search
        classifier_search = hyper_search.hyper_search(classifier, input_set, target_set, deep=deep)

        # save model and prediction result
        Parameter_path = f"Model_parameters/{Data_version}"
        Save_model(classifier, input_set, target_set, path=Parameter_path)

        return classifier_search


if __name__ == '__main__':

    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    # define type of Model
    model_type = "Regressor"
    model_type = "Classifier"

    # optimize hyper parameter
    deep_space = np.linspace(1, 1, num=1)
    for deep in deep_space:
        parameter_space = optimize(model_type, deep=deep)

        # train MLP
        main(model_type, parameter_space)