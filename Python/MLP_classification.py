import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Evaluation import plot_learning_curve as plc
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing
from warnings import simplefilter

def dataset_reader(path='Data/daten', name='1P1K', type='csv'):
    """
    read the data set
    and return Input and Target of Training Network
    :param path: [str], the path of csv data (default value : 'Data/rt-daten')
    :param name: [str], data to import ('1P' ,'1P1K' or '1PmitT', e.g.) (default value : '1P1K')
    :param type: [str], data type (default value : 'csv')
    :return input_set: [narray],  Input data set
    :return target_set: [narray], Target data set
    """
    # definition the data version
    datasets_v1 = ['1P', '2p', '3P', '4P', '5P', '6P', '7P']
    datasets_v2 = ['1P1K', '2P1K', '3P1K', '4P1K', '5P1K', '6P1K', '7P1K']
    datasets_v3 = ['1PmitT', '2PmitT', '3PmitT', '4PmitT', '5PmitT', '6PmitT', '7PmitT']

    # assign the data path, prepare for loading
    data = f"{path}{name}.{type}"
    # load the csv data with pandas
    df = pd.read_csv(data)

    # assign Input label
    label_x = ['omega_1', 'omega_2', 'omega_3', 'D_1', 'D_2', 'D_3', 'EVnorm1_1', 'EVnorm1_2', 'EVnorm1_3',
               'EVnorm2_1', 'EVnorm2_2', 'EVnorm2_3', 'EVnorm3_1', 'EVnorm3_2', 'EVnorm3_3']

    # switch Target label based on data version
    if name in datasets_v1:
        label_y = ['m2', 'm3', 'm4', 'k5', 'k6', 'alpha', 'beta']
    elif name in datasets_v2:
        label_y = ['m2', 'm3', 'm4', 'k5plusk6', 'alpha', 'beta']
    elif name in datasets_v3:
        label_y = ['m2', 'm3', 'm4', 'k', 'alpha', 'beta']
        label_x.append('Tem')


    input_set = df[label_x].values
    target_set = df[label_y].values

    return input_set, target_set


def dataset_preprocess(input_set, target_set=None):
    """
    regularise Input data set

    :param input_set: [narray],  Input data set

    :return input_set: [narray],  Input data set after regularization
    :return input_std: [narray],  regularization standard deviation
    :return input_mean: [narray],  regularization mean
    """
    # regularization Input
    input_set = preprocessing.scale(input_set)

    # compute feature scaling parameter
    input_std = np.std(input_set, ddof=0)  # standard deviation with bias (column)
    input_mean = input_set.mean(0)  # mean (column)

    # assign binary mask to Target classification
    if target_set is not None:
        # assign reference value
        target_set_ref = [1, 1, 1, 2, 0.6261, 0.0001]
        # decide change of target parameter
        for i, reference in enumerate(target_set_ref):
            target_set[:, i] = target_set[:, i] != reference
        # convert boolen to int
        target_set = target_set.astype(int)

        # convert [5*1] array in [1*1] float
        for i in range(target_set.shape[1]):
            target_set[:, i] = target_set[:, i] * 10**i
        target_set = np.sum(target_set, axis=1, dtype=int)

        return input_set, input_std, input_mean, target_set

    return input_set, input_std, input_mean


def regression(input_set, target_set, test_size=0.2, random_seed=23, alpha=8e-3, hidden_layer_sizes=(105, 70, 46), max_iter=500):
    """
    modeling a MLP Regressor with random split all data set.
    after training print out test score on console.

    :param input_set: [narray],  Input of Training Network
    :param target_set: [narray],  Target of Training Network
    :param test_size: [float], the proportion of test data in all data set (default value : 0.2)
    :param random_seed: [int], the random seed of random split for data set (default value : 233)
    :param alpha: [float], regularisation coefficient in MLP Regressor (default value : 8e-3)
    :param hidden_layer_sizes: [tuple of int], structural hyperparameter in MLP Regressor (default value : (105, 70, 46))
    :param max_iter: [int], maximal iteration epoch in MLP Regressor (default value : (105, 70, 46))

    :return regressor: [estimator],  MLP Regressor with
    :return score: [float], determination coefficient of test data set
    :return weight_matrix: [narray], weight matrix of training data set
    """

    # split into training and test set
    X_train, X_test, y_train, y_test = \
        train_test_split(input_set, target_set, test_size=test_size, random_state=random_seed)

    # setup a MLP Regressor 3 layers
    regressor = MLPRegressor(solver='lbfgs', alpha=alpha,
                       hidden_layer_sizes=hidden_layer_sizes, random_state=1, max_iter=max_iter)

    # fit Regressor to the training data
    regressor.fit(X_train, y_train)

    # compute and Print R2 Metrics
    score = regressor.score(X_test, y_test)
    print('Test R2 Score: %f' % score)

    # compute Weight matrix
    weight_matrix = regressor.fit(X_train, y_train).coefs_

    return regressor, score, weight_matrix


def classifier(input_set, target_set, test_size=0.2, random_seed=23, alpha=8e-3, hidden_layer_sizes=(105, 70, 46), max_iter=500):
    """
    modeling a MLP classifier with random split all data set.
    after training print out test score on console.

    :param input_set: [narray],  Input of Training Network
    :param target_set: [narray],  Target of Training Network
    :param test_size: [float], the proportion of test data in all data set (default value : 0.2)
    :param random_seed: [int], the random seed of random split for data set (default value : 233)
    :param alpha: [float], regularisation coefficient in MLP Regressor (default value : 8e-3)
    :param hidden_layer_sizes: [tuple of int], structural hyperparameter in MLP Regressor (default value : (105, 70, 46))
    :param max_iter: [int], maximal iteration epoch in MLP Regressor (default value : (105, 70, 46))

    :return regressor: [estimator],  MLP Regressor with
    :return score: [float], accuracy of test data set
    :return weight_matrix: [narray], weight matrix of training data set
    """

    # split into training and test set
    X_train, X_test, y_train, y_test = \
        train_test_split(input_set, target_set, test_size=test_size, random_state=random_seed)

    # setup a MLP Classifier 3 layers
    classifier = MLPClassifier(solver='lbfgs', alpha=alpha, activation='logistic',
                       hidden_layer_sizes=hidden_layer_sizes, random_state=1, max_iter=max_iter)

    # fit Regressor to the training data
    classifier.fit(X_train, y_train)

    # compute and Print R2 Metrics
    score = classifier.score(X_test, y_test)
    print('Test Score: %f' % score)

    return classifier, score


def confusion_matrix(estimator, input_set, target_set, random_seed=23, test_size=0.2):
    """
        modeling a MLP classifier with random split all data set.
        after training print out test score on console.

        :param input_set: [narray],  Input of Training Network
        :param target_set: [narray],  Target of Training Network
        :param test_size: [float], the proportion of test data in all data set (default value : 0.2)
        :param random_seed: [int], the random seed of random split for data set (default value : 233)

        :return regressor: [estimator],  MLP Regressor with
        :return score: [float], accuracy of test data set
        :return weight_matrix: [narray], weight matrix of training data set
        """
    # Split into training and test set
    X_train, X_test, y_train, y_test = \
        train_test_split(input_set, target_set.astype(int), test_size=test_size, random_state=random_seed)

    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = estimator.fit(X_train, y_train)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     # display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()


if __name__ == '__main__':
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    # load data and preprocess
    input_set, target_set = dataset_reader(name='2PmitT')
    input_set, _, _, target_set = dataset_preprocess(input_set, target_set)  # preprocess for classification

    # model MLP Classifie
    MLP_classifier, _ = classifier(input_set, target_set)

    # evaluation
    confusion_matrix(MLP_classifier,input_set,target_set)