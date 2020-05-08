import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix


def confusion_matrix(estimator, input_set, target_set, target_name, random_seed=23, test_size=0.2):
    """
        modeling a MLP classifier with random split all data set.
        after training print out test score on console.

        :param input_set: [narray],  Input of Training Network
        :param target_set: [narray],  Target of Training Network
        :param test_size: [float], the proportion of test data in all data set (default value : 0.2)
        :param random_seed: [int], the random seed of random split for data set (default value : 23)

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
                                     display_labels=target_name,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()