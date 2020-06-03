import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=(20, 5))  # Initialize subplot

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim) # set default ylim (automatic matching)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    # evaluate the model quality
    # train_sizes: Numbers of training examples that has been used to generate the learning curve.
    # train_scores：Scores on training sets, R2
    # test_scores: Scores on test set, R2
    # fit_times: Times spent for fitting (in seconds)
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    # estimator: Model object, e.g. ANN, SVM, etc
    # X: Training vector
    # y: Target relative to X
    # cv: cross-validation splitting strategy.
    # train_sizes: Relative numbers of training examples that will be used to generate the learning curve
    # n_jobs: Number of processors to run in parallel. None means 1 processors
    # return_times: Whether to return the fit and score times.

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


def evaluation_learning_curve(estimator, input_set, target_set,
                              title="(46, 29, 26)", test_size=0.2, train_sizes=np.linspace(0.1, 1.0, 20)):
    """
    evaluate estimator fitting quality and performance, which implements on random cross validation.
    the data size of cross validation will linear increase, so that could research the overfitting and underfitting problem
    every cross validation is isolated, each data set dosen't have any influence of any other data sets.
    after each cross validation cache the score and time cost, finally will be visualized.

    :param estimator: [estimator],  MLP Perceptron model
    :param input_set: [narray],  Input data set
    :param target_set: [narray],  Target data set
    :param title: [str], string of structural hyperparameter in MLP Perceptron (default value : "(105, 70, 46)")
    :param test_size: [float], the proportion of test data in all data set (default value : 0.2)
    :param train_sizes: [nrarray], the proportion of cross validation data in all data set (default value : np.linspace(0.01, 1.0, 25))
    """
    # set the Title of Learning curve
    title = "Learning Curves" + title

    # randomly select cross validation set.
    cv = ShuffleSplit(test_size=test_size)

    # recall function
    plot_learning_curve(estimator, title, input_set, target_set,
                            cv=cv, ylim=(0.73, 1.01), n_jobs=-1, train_sizes=train_sizes)
    plt.show()
    plt.savefig('Model_parameters/version_6/LearningCurve.pdf')
