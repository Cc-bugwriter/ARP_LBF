
"""
this script imports the csv data from the FE model in MATLAB of the shape of
m2,m3,m4,k5,k6,alpha,beta,omega_1,omega_2,omega_3,D_1,D_2,D_3,EVnorm1_1,EVnorm1_2,EVnorm1_3,EVnorm2_1,EVnorm2_2,EVnorm2_3,EVnorm3_1,EVnorm3_2,EVnorm3_3
and saves it as pickle file for short loading times.
"""

import pandas as pd
import pickle

def preprocess(dataset,comment=''):
    """
    Imports modal parameter data from FEM data exported from MATLAB, splits it into train
    and test data and saves it as pickle files .train and .test in the data directory
    
    :param dataset: [int] Number of the data set preprocessed. Refer to 'Daten_Erklärung.doc'
    :param comment: [str] comment that is attached to the end of the file name
    :return:
    """

    train_path = f"Data/daten{dataset}P_{comment}.train"
    test_path = f"Data/daten{dataset}P_{comment}.test"

    df = pd.read_csv(f"Data/daten{dataset}P.csv")

    n_rows = df.shape[0]
    n_split = int(0.7*n_rows)

    df_train = df.iloc[:n_split]
    df_test = df.iloc[n_split:n_rows]

    df_train.to_pickle(train_path)
    df_test.to_pickle(test_path)    # information about pickling under https://www.youtube.com/watch?v=2Tw39kZIbhs 4:50 how to import


if __name__ == '__main__':

    data_set = 1    # dataset to load, refer to 'Daten_Erklärung.doc'
    comment = ''    # attached to the end of the filename

    preprocess(data_set,comment=comment)