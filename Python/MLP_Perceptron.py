import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Preprocessing import pre_processing
from Regression import Regressor
from Classification import Classifier
from Evaluation import plot_learning_curve as plc
from warnings import simplefilter


def main(model):
    """
    main function of NN
    :param model: [boolen],  MLP perceptron model ("Classifier" or "Regressor")
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
        regressor, score, weight_matrix = Regressor.regression(input_set, target_set)
    elif model == "Classifier":
        classifier, score, weight_matrix = Classifier.classifier(input_set, target_set)

    # evaluate MLP preceptron


if __name__ == '__main__':
    main("Regressor")