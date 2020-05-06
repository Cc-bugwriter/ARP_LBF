import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_learning_curve as plc
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Import data
df = pd.read_csv('daten1P1K.csv')
label_y = ['m2', 'm3', 'm4', 'k5plusk6', 'alpha', 'beta']
label_x = ['omega_1', 'omega_2', 'omega_3', 'D_1', 'D_2', 'D_3', 'EVnorm1_1', 'EVnorm1_2', 'EVnorm1_3',
           'EVnorm2_1', 'EVnorm2_2', 'EVnorm2_3', 'EVnorm3_1', 'EVnorm3_2', 'EVnorm3_3']

y = df[label_y].values
X = df[label_x].values

# Regularization Input
X = preprocessing.scale(X)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Setup a ANN Regressor 2 layers
knn = MLPRegressor(solver='lbfgs', alpha=8e-3,
                    hidden_layer_sizes=(105,70,46), random_state=1, max_iter=500)
# Fit the Regressor to the training data
knn.fit(X_train, y_train)
# Print R2 Metrics
print(knn.score(X_test, y_test))

Gewichtsmatrix = knn.fit(X, y).coefs_

# Evaluation R2, Fit-time
estimator = MLPRegressor(solver='lbfgs', alpha=8e-3,
                    hidden_layer_sizes=(105, 70, 46), random_state=1)
# Set the Title of Learning curve
title = "Learning Curves (105, 70, 46)"
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(test_size=0.2)
# Set train_sizes
train_sizes=np.linspace(0.01, 1.0, 25)
plc.plot_learning_curve(estimator, title, X, y, cv= cv, ylim=(0.,1.01), n_jobs=6, train_sizes=train_sizes)
plt.show()

# Compute train and test errors
alphas = np.logspace(-5, -2, 30)
train_errors = list()
test_errors = list()
for alpha in alphas:
    knn.set_params(alpha=alpha)
    knn.fit(X_train, y_train)
    train_errors.append(knn.score(X_train, y_train))
    test_errors.append(knn.score(X_test, y_test))

i_alpha_optim = np.argmax(test_errors)
alpha_optim = alphas[i_alpha_optim]
print("Optimal regularization parameter : %s" % alpha_optim)

# Estimate the coef_ on full data with optimal regularization parameter
knn.set_params(alpha=alpha_optim)
knn_loss = knn.fit(X, y).loss_
print(knn_loss)

plt.semilogx(alphas, train_errors, label='Train')
plt.semilogx(alphas, test_errors, label='Test')
plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
           linewidth=3, label='Optimum on test')
plt.legend(loc='lower left')
plt.ylim([0.95, 1.01])
plt.xlabel('Regularization parameter')
plt.ylabel('Performance')
plt.legend()
plt.show()