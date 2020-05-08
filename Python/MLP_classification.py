import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing
from warnings import simplefilter


class Classification():
    """
    define all classification with enumeration
    """
    m2 = 1
    m3 = 10
    m4 = 100
    k = 1000
    a = 10000
    b = 100000
    belong_to = \
        {0: 'none',
          m2: 'm2', m3: 'm3', m4: 'm4', k: 'k', a: 'alpha', b: 'beta',
          m2 + m3: 'm23', m2 + m4: 'm24', m2 + k: 'm2k', m2 + a: 'm2a', m2 + b: 'm2b',
          m3 + m4: 'm34', m3 + k: 'm3k', m3 + a: 'm3a', m3 + b: 'm3b',
          m4 + k: 'm4k', m4 + a: 'm4a', m4 + b: 'm4b',
          k + a: 'ka', k + b: 'kb',
          a + b: 'ab',
          m2 + m3 + m4: 'm234', m2 + m3 + k: 'm23k', m2 + m3 + a: 'm23a', m2 + m3 + b: 'm23b', m2 + m4 + k: 'm24k',
          m2 + m4 + a: 'm24a', m2 + m4 + b: 'm24b', m2 + k + a: 'm2ka', m2 + k + b: 'm2kb', m2 + a + b: 'm2ab',
          m3 + m4 + k: 'm34k', m3 + m4 + a: 'm34a', m3 + m4 + b: 'm34b', m3 + k + a: 'm3ka', m3 + k + b: 'm3kb',
          m3 + a + b: 'm3ab',
          m4 + k + a: 'm4ka', m4 + k + b: 'm4kb', m4 + a + b: 'm4ab',
          k + a + b: 'kab',
          m2 + m3 + m4 + k: 'm234k', m2 + m3 + m4 + a: 'm234a', m2 + m3 + m4 + b: 'm234b', m2 + m3 + k + a: 'm23ka',
          m2 + m3 + k + b: 'm23kb', m2 + m3 + a + b: 'm23ab', m2 + m4 + k + a: 'm24ka', m2 + m4 + k + b: 'm24kb',
          m2 + m4 + a + b: 'm24ab', m2 + k + a + b: 'm2kab',
          m3 + m4 + k + a: 'm34ka', m3 + m4 + k + b: 'm34kb', m3 + m4 + a + b: 'm34ab', m3 + k + a + b: 'm3kab',
          m4 + k + a + b: 'm4kab',
          m2 + m3 + m4 + k + a: 'm234ka', m2 + m3 + m4 + k + b: 'm234kb', m2 + m3 + m4 + a + b: 'm234ab',
          m2 + m3 + k + a + b: 'm23kab', m2 + m4 + k + a + b: 'm24kab', m3 + m4 + k + a + b: 'm34kab',
          m2 + m3 + m4 + k + a + b: 'm234kab'}


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
    :param target_set: [narray],  Target data set (default value: None)

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

        # convert [1*1] float to str
        target_set_list = target_set.tolist()
        target_name = []
        for i in target_set_list:
            target_name.append(Classification.belong_to[i])
        target_name = np.array(target_name)

        return input_set, input_std, input_mean, target_set, target_name

    return input_set, input_std, input_mean


def classifier(input_set, target_set, test_size=0.2, random_seed=23, alpha=0.00013738, hidden_layer_sizes=(46, 29, 26), max_iter=1000):
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

    :return regressor: [estimator],  MLP Classifier with
    :return score: [float], accuracy of test data set
    :return weight_matrix: [narray], weight matrix of training data set
    """

    # split into training and test set
    X_train, X_test, y_train, y_test = \
        train_test_split(input_set, target_set, test_size=test_size, random_state=random_seed)

    # setup a MLP Classifier 3 layers
    classifier = MLPClassifier(solver='lbfgs', alpha=alpha,
                       hidden_layer_sizes=hidden_layer_sizes, random_state=1, max_iter=max_iter)

    # fit Regressor to the training data
    classifier.fit(X_train, y_train)

    # compute and Print R2 Metrics
    score = classifier.score(X_test, y_test)
    print('Test Score (Accuracy): %f' % score)

    # compute Weight matrix
    weight_matrix = classifier.fit(X_train, y_train).coefs_

    return classifier, score, weight_matrix


def confusion_matrix(estimator, input_set, target_set, target_name, random_seed=23, test_size=0.2):
    """
        modeling a MLP classifier with random split all data set.
        after training print out test score on console.

        :param input_set: [narray],  Input of Training Network
        :param target_set: [narray],  Target of Training Network
        :param test_size: [float], the proportion of test data in all data set (default value : 0.2)
        :param random_seed: [int], the random seed of random split for data set (default value : 23)

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
                                     display_labels=target_name,
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
    for i in range(1, 7):
        name = f"{i}PmitT"
        input_set, target_set = dataset_reader(name=name)
        input_set, _, _, target_set, target_name = dataset_preprocess(input_set, target_set)
        MLP_classifier, _, _ = classifier(input_set, target_set)

    # preprocess for classification
    input_set, _, _, target_set = dataset_preprocess(input_set, target_set)

    # model MLP Classifie
    MLP_classifier, _, _ = classifier(input_set, target_set)

    # evaluation
    confusion_matrix(MLP_classifier, input_set, target_set, target_name)