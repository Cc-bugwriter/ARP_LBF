import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Preprocessing import pre_processing
from Processing import Regressor, Classifier, Save_model
from Evaluation import plot_learning_curve
from Evaluation import confusion_matrix
from Optimation import hyper_search
from sklearn.neural_network import MLPRegressor, MLPClassifier


def main(model_type, hyperparameter=None, data_version="version_6", evaluation=False, first_loc=1, end_loc=7):
    """
    main function of MLP
    :param model_type: [str], MLP perceptron model ("Classifier" or "Regressor")
    :param hyperparameter: [dict], result of hyper search
    :param data_version: [str], version of data set ('version_1', 'version_2', 'version_3', e.g.),
    (default value: "version_4")
    :param evaluation: [boolean], determination, whether evaluate the fitting process or not
    :param first_loc:  [int], first data set index (default value: 1)
    :param end_loc:  [int], end data set index (default value: 7)
    """
    # assign parameter save and load path
    parameter_path = f"Model_parameters/{data_version}"

    if model_type == "Regressor":
        # preprocess for MLP preceptron
        X_train, y_train, X_del, y_del, X_test, y_test = \
            pre_processing.merge_split(data_version="version_6", first_loc=first_loc, end_loc=end_loc)

        # training MLP preceptron
        regressor = Regressor.regression(X_train, y_train, X_test, y_test,
                                         hyperparameter=hyperparameter, version=data_version)

        # save model and prediction result
        Save_model.save_Preceptron(regressor, X_test, y_test, path=parameter_path, overwrite=True)

        # evaluate fitting process
        if evaluation:
            plot_learning_curve.evaluation_learning_curve(regressor, X_del, y_del,
                                                          title=f"{regressor.get_params()['hidden_layer_sizes']}")

    elif model_type == "Classifier":
        # preprocess for MLP preceptron
        X_train, y_train, X_del, y_del, X_test, y_test = \
            pre_processing.merge_split(data_version="version_6", first_loc=first_loc, end_loc=end_loc, regressor=False)

        # training MLP preceptron
        classifier = Classifier.classifier(X_train, y_train, X_test, y_test, hyperparameter=hyperparameter)

        # save model and prediction result
        Save_model.save_Preceptron(classifier, X_test, y_test, path=parameter_path)

        # evaluate MLP preceptron
        if evaluation:
            confusion_matrix.confusion_matrix(classifier, X_test, y_test, target_name=None)


def optimize(model, deep=3, data_version="version_6", first_loc=1, end_loc=7):
    """
    optimize function of MLP
    :param model: [str],  MLP perceptron model ("Classifier" or "Regressor")
    :param deep: [int], depth of MLP Network  (default value: 3)
    :param data_version: [str], version of data set ('version_1', 'version_2', 'version_3', e.g.),
     (default value: "version_6")
    :param first_loc:  [int], first data set index (default value: 1)
    :param end_loc:  [int], end data set index (default value: 7)
    :param first_loc:  [int], first data set index (default value: 1)
    :param end_loc:  [int], end data set index (default value: 7)

    :return para_space: [dict], MLP Hyper parameter
    """
    if model == "Regressor":
        # preprocess for MLP preceptron
        _, _, X_del, y_del, _, _ = \
            pre_processing.merge_split(data_version="version_6", first_loc=first_loc, end_loc=end_loc)

        # setup a MLP preceptron
        regressor = MLPRegressor(solver='lbfgs', random_state=1)

        # implement hyper search
        regressor_search = hyper_search.hyper_search(regressor, X_del, y_del, deep=deep, version=data_version)

        return regressor_search

    elif model == "Classifier":
        # preprocess for MLP preceptron
        _, _, X_del, y_del, _, _ = \
            pre_processing.merge_split(data_version="version_6", first_loc=first_loc, end_loc=end_loc, regressor=False)

        # setup a MLP preceptron
        classifier = MLPClassifier(solver='lbfgs', random_state=1)

        # implement hyper search
        classifier_search = hyper_search.hyper_search(classifier, X_del, y_del, deep=deep, version=data_version)

        return classifier_search


if __name__ == '__main__':
    # parameter space (control)
    first_loc = 1
    end_loc = 1
    data_version = "version_6"
    evaluation = False
    opt = False

    # define type of Model
    model_type = "Regressor"
    # model_type = "Classifier"

    if not opt:
        main(model_type, data_version="version_6", evaluation=evaluation, first_loc=first_loc, end_loc=end_loc)
    else:
        # optimize hyper parameter
        deep_space = np.linspace(1, 3, num=1)
        for deep in deep_space:
            parameter_space = optimize(model_type, deep=int(deep))
            # train MLP
            main(model_type, parameter_space)
            if deep == 3:
                main(model_type, parameter_space, evaluation=evaluation, first_loc=first_loc, end_loc=end_loc)

    # # full path： "Model_parameters/version_4/classifier_layer_1.joblib"
