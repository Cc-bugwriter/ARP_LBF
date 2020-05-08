import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Preprocessing import pre_processing
from Regression import Regressor
from Classification import Classifier
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
    input_set, target_set = pre_processing.merge_data()

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


def optimize(model, deep_space=np.linspace(3, 3, num=1)):
    """
    optimize function of NN
    :param model: [str],  MLP perceptron model ("Classifier" or "Regressor")
    :param deep_space: [narray],  network layer space (default value: np.linspace(3, 3, num=1))
    """
    # load data set
    input_set, target_set = pre_processing.merge_data()

    # preprocess for MLP preceptron
    if model == "Regressor":
        input_set, _, _ = pre_processing.dataset_preprocess(input_set)
    elif model == "Classifier":
        input_set, _, _, target_set, target_name = pre_processing.dataset_preprocess(input_set, target_set)

    # setup a MLP preceptron 3 layers
    if model == "Regressor":
        regressor = MLPRegressor(solver='lbfgs', random_state=1)
    elif model == "Classifier":
        classifier = MLPClassifier(solver='lbfgs', random_state=1)

    # search best hyper parameter
    if model == "Regressor":
        regressor_search = hyper_search.hyper_search(regressor, input_set, target_set)
        return regressor_search
    elif model == "Classifier":
        classifier_search = hyper_search.hyper_search(classifier, input_set, target_set)
        return classifier_search


if __name__ == '__main__':

    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    # optimize hyper parameter
    parameter_space = optimize("Classifier")

    # # train network
    # main("Classifier", parameter_space)