"""
This script loads the preprocessed pickled data sets, trains a model based on KNN,
evaluates it using the error metrics R2 value and MAE, and then saves the model
as well as the evaluation information.
"""
import os
import pandas as pd
from pathlib import Path
import pickle
import multiprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
import time
# import chosen KNN library

#set up
comment = ''
dataset = 1


def data_set_loader(dataset, type='train',comment=''):
    """
    Function to load the preprocessed pickled data set. Returns the data set as pandas dataframe.

    :param dataset: [int] Number of the data set preprocessed. Refer to 'Daten_Erklärung.doc'
    :param type:[str] data to import ('train' or 'test') (default value : 'train')
    :param comment:[str/int] comment at the end of the file (default value: '')
    :return: [panda.Dataframe]
    """

    train_path = f"Data/daten{dataset}P_{comment}.{type}"
    df = pd.read_pickle(train_path)
    y_labels = ['m2','m3','m4','k5','k6','alpha','beta']
    return df.drop(columns=y_labels) , df[y_labels]


def save_model(object_, dataset, comment=''):
    """
    Saves the trained model in directory 'models'

    :param object_: Model to save
    :param dataset:[int] the number of the dataset
    :param comment: [str] comment to add to the file name (optional, default: '')
    :return: None
    """

    model_path = f"models/dataset{dataset}_{comment}.model"
    with open(model_path,'wb') as f:
        pickle.dump(object_, f)


def modelling_data(y_pred, y_test, dataset, training_time=0):
    """
    To calculate error for a trained model and save it into the corresponding txt files.

    :param y_pred: [pandas.Series] predicted values by the model
    :param y_test: [pandas.Series] true values
    :param dataset:[int] the number of the dataset
    :param training_time: [int] training time of the model in seconds (default value: 0)
    :return: None
    """

    filepath = Path(f"./models/dataset{dataset}_.txt")    #Filename
    fieldnames = ['dataset;', 'MAE (add UNITS here);', 'R2;', 'Train Time']   #Columns

    if not (os.path.isfile(filepath)):          #if the file doesn't already exist we need to write the columns name
        header=True
    else:
        header=False

    with open(filepath,'a') as f:

        if header:
            f.writelines(fieldnames)
            f.write("\n")

        error_rate_mae = mean_absolute_error(y_test, y_pred)        #calculate mean average error
        error_rate_r2 = r2_score(y_test, y_pred)                    #calculete the r2 score
        f.writelines([str(dataset),";", str(error_rate_mae),";", str(error_rate_r2),";", str(training_time)])
        f.write("\n")


if __name__ == '__main__':

    print(f"Start training model on dataset{dataset}")

    # Data import
    x_train, y_train = data_set_loader(dataset, comment=comment)

    # Training
    model = pipeline.make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(50, 30, 15),
                                                                    max_iter=150, random_state=0, tol=0.01, verbose=0))
    start_time = time.time()
    model.fit(x_train, y_train)  # Training of the model
    train_time = time.time() - start_time

    print(f"End training dataset{dataset} in {train_time}s")
    save_model(model, dataset, comment='')

    # Evaluation
    x_test, y_test = data_set_loader(dataset, type="test", comment=comment)
    y_pred = model.predict(x_test)
    modelling_data(y_pred, y_test, dataset, training_time=train_time)