import pandas as pd
import numpy as np
from sklearn.preprocessing import add_dummy_feature


def dataset_reader(path='Data', data_version='version_6', name='1P', type='csv'):
    """
    read the data set
    and return Input and Target of Training Network
    :param path: [str], the path of csv data (default value : 'Data')
    :param data_version: [str], the version of csv data (default value : 'version_4')
    :param name: [str], data to import ('1P' ,'1P1K' or '1PmitT', e.g.) (default value : '1P')
    :param type: [str], data type (default value : 'csv')
    :return input_set: [narray],  Input data set
    :return target_set: [narray], Target data set
    """
    # assign the data path, prepare for loading
    data = f"{path}/{data_version}/daten{name}.{type}"
    # load the csv data with pandas
    df = pd.read_csv(data)

    # assign Input label
    label_x = ['omega_1', 'omega_2', 'omega_3', 'D_1', 'D_2', 'D_3', 'EVnorm1_1', 'EVnorm1_2', 'EVnorm1_3',
               'EVnorm2_1', 'EVnorm2_2', 'EVnorm2_3', 'EVnorm3_1', 'EVnorm3_2', 'EVnorm3_3']

    # switch Target label based on data version
    if data_version == "version_1":
        label_y = ['m2', 'm3', 'm4', 'k5', 'k6', 'alpha', 'beta']
    elif data_version == "version_2":
        label_y = ['m2', 'm3', 'm4', 'k5plusk6', 'alpha', 'beta']
    else:
        label_y = ['m2', 'm3', 'm4', 'k', 'alpha', 'beta']
        label_x.append('Tem')

    input_set = df[label_x].values
    target_set = df[label_y].values

    return input_set, target_set


def merge_data(data_version="version_6", first_loc=1, end_loc=7):
    """
    merge all data in a couple of data sets
    :param data_version: [str], version of data set ('version_1', 'version_2', 'version_3', e.g.)
    (default value: "version_5")
    :param first_loc:  [int], first data set index (default value: 1)
    :param end_loc:  [int], end data set index (default value: 7)
    :return input_set: [narray],  Input data set
    :return target_set: [narray], Target data set
    """
    # define version list
    version_list = {"version_1": 'P', "version_2": 'P1K', "version_3": 'PmitT',
                    "version_4": 'P', "version_5": 'P_gerundet', "version_6": 'P',
                    "version_7": '_rauschen', "version_8": '_rauschen_sigma',
                    "version_9": 'P_grosserBereich'}

    # assign data name from version
    data_name = version_list[data_version]

    input_set, target_set = dataset_reader(data_version=data_version, name=f'{first_loc}{data_name}')

    data_len = end_loc+1

    for i in range(first_loc+1, data_len):
        name = f"{i}{data_name}"
        input_append, target_append = dataset_reader(data_version=data_version, name=name)
        input_set = np.concatenate((input_set, input_append), axis=0)
        target_set = np.concatenate((target_set, target_append), axis=0)

    return input_set, target_set