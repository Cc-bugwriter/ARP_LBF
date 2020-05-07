from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


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

    :return regressor: [estimator],  MLP Regressor with
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

    # compute aWeight matrix
    weight_matrix = classifier.fit(X_train, y_train).coefs_

    return classifier, score, weight_matrix