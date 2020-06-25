import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from Preprocessing import dataset_reader as dr
from sklearn.model_selection import train_test_split


class Classification():
    """
    define all classification with enumeration
    """
    # assign number mask for classification
    m2 = 1
    m3 = 10
    m4 = 100
    k = 1000
    a = 10000
    b = 100000

    # define class dictionary
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
        m2 + m3 + m4 + k + a + b: 'm234kab'
         }


def data_scaling(input_set, version="version_6"):
    """
    regularise input data set

    :param input_set: [narray],  Input data set
    :param version: [str], version of data set ('version_1', 'version_2', 'version_3', e.g.),
     (default value: "version_6")

    :return scaled_set: [narray],  Input data set after regularization
    """
    # assign scaler path
    scaler_name = "scaler.joblib"
    directory_path = f"Model_parameters/{version}"
    scaler_path = f"Model_parameters/{version}/{scaler_name}"

    # initial StandardScaler
    scaler = StandardScaler()

    # determine path
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    if not os.path.exists(scaler_path):
        all_data, _ = dr.merge_data(version, first_loc=1, end_loc=7)
        scaler.fit(all_data)

        # save the scaler at first fitting
        joblib.dump(scaler, scaler_path)
        print("save scaler successfully")
    else:
        # direct read scaler
        scaler = joblib.load(scaler_path)
        print("load scaler successfully")

    scaled_set = scaler.transform(input_set)
    return scaled_set


def dataset_preprocess(input_set, target_set=None, data_version="version_6"):
    """
    regularise Input data set
    return target class for classifier

    :param input_set: [narray],  Input data set
    :param target_set: [narray],  Target data set (default value: None)
    :param data_version: [str], version of data set ('version_1', 'version_2', 'version_3', e.g.),
     (default value: "version_6")

    :return input_set: [narray],  Input data set after regularization
    :return input_std: [narray],  regularization standard deviation
    :return input_mean: [narray],  regularization mean
    """
    # # compute feature scaling parameter
    # input_std = np.std(input_set, ddof=0)  # standard deviation with bias (column)
    # input_mean = input_set.mean(0)  # mean (column)

    # regularization Input
    input_set = data_scaling(input_set, version=data_version)

    # assign binary mask to Target classification [labeling]
    if target_set is not None:
        # assign reference value
        # target_set_ref = [1, 1, 1, 2, 0.6261, 0.0001]  # old version
        target_set_ref = [1, 1, 1, 2, 0.6261, 1]
        # decide change of target parameter
        for i, reference in enumerate(target_set_ref):
            target_set[:, i] = target_set[:, i] != reference

        # convert boolen to int
        target_set = target_set.astype(int)

        # convert [5*1] array to [1*1] float  [labeling]
        for i in range(target_set.shape[1]):
            target_set[:, i] = target_set[:, i] * 10 ** i
        target_set = np.sum(target_set, axis=1, dtype=int)

        # convert [1*1] float to str [labeling]
        target_set_list = target_set.tolist()

        target_name = []
        for i in target_set_list:
            target_name.append(Classification.belong_to[i])
        target_name = np.array(target_name)

        return input_set, target_set, target_name

    return input_set


def single_data_split(input_set, target_set, test_size=0.2, random_seed=233):
    """
    input_set as a single data set
    split input_set as training, development and test data sets
    :param input_set: [narray], Input data set
    :param target_set: [narray], Target data set (default value: None)

    :return X_train: [narray], training data input
    :return y_train: [narray], training data target
    :return X_del: [narray], development data input
    :return y_del: [narray], development data target
    :return X_test: [narray], test data input
    :return y_test: [narray], test data target
    """
    # split into training and test set
    X_train_, X_test, y_train_, y_test = \
        train_test_split(input_set, target_set, test_size=test_size, random_state=random_seed)

    # split into training set with development set
    X_train, X_del, y_train, y_del = \
        train_test_split(X_train_, y_train_, test_size=test_size/(1-test_size), random_state=random_seed)

    return X_train, y_train, X_del, y_del, X_test, y_test


def merge_split(data_version="version_6", first_loc=1, end_loc=7, regressor=True):
    """
    merge all data in a couple of data sets
    :param data_version: [str], version of data set ('version_1', 'version_2', 'version_3', e.g.)
    (default value: "version_5")
    :param first_loc:  [int], first data set index (default value: 1)
    :param end_loc:  [int], end data set index (default value: 7)
    :param regressor:  [boolean], Estimator mode (True, False), (default value: True)

    :return X_train: [narray], training data input
    :return y_train: [narray], training data target
    :return X_del: [narray], development data input
    :return y_del: [narray], development data target
    :return X_test: [narray], test data input
    :return y_test: [narray], test data target
    """
    # define version list
    version_list = {"version_1": 'P', "version_2": 'P1K', "version_3": 'PmitT',
                    "version_4": 'P', "version_5": 'P_gerundet', "version_6": 'P',
                    "version_7": '_rauschen', "version_8": '_rauschen_sigma',
                    "version_9": 'P_grosserBereich'}
    # assign data name from version
    data_name = version_list[data_version]

    # load first data set
    input_set, target_set = dr.dataset_reader(data_version=data_version, name=f'{first_loc}{data_name}')

    # split first data set
    X_train, y_train, X_del, y_del, X_test, y_test = \
        single_data_split(input_set, target_set)

    data_len = end_loc + 1

    for i in range(first_loc+1, data_len):
        name = f"{i}{data_name}"
        # load next data set
        input_set, target_set = dr.dataset_reader(data_version=data_version, name=name)

        # split next data set
        X_train_append, y_train_append, X_del_append, y_del_append, X_test_append, y_test_append = \
            single_data_split(input_set, target_set)

        # concatenate the data set (Merge data section)
        X_train = np.concatenate((X_train, X_train_append), axis=0)
        y_train = np.concatenate((y_train, y_train_append), axis=0)
        X_del = np.concatenate((X_del, X_del_append), axis=0)
        y_del = np.concatenate((y_del, y_del_append), axis=0)
        X_test = np.concatenate((X_test, X_test_append), axis=0)
        y_test = np.concatenate((y_test, y_test_append), axis=0)

    # scaling concatenation data
    if regressor:
        # regressor mode
        X_train = dataset_preprocess(X_train, data_version=data_version)
        X_del = dataset_preprocess(X_del, data_version=data_version)
        X_test = dataset_preprocess(X_test, data_version=data_version)

        return X_train, y_train, X_del, y_del, X_test, y_test

    else:
        # classifier mode
        X_train, y_train, _ = dataset_preprocess(X_train, y_train, data_version=data_version)
        X_del, y_del, _ = dataset_preprocess(X_del, y_del, data_version=data_version)
        X_test, y_test, _ = dataset_preprocess(X_test, y_test, data_version=data_version)

        return X_train, y_train, X_del, y_del, X_test, y_test
