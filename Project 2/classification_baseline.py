from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

# Gets data
from read_ecoli_data import *

# Set cross-validation parameters
K1 = 10
K2 = K1
CV = model_selection.KFold(n_splits=K1,shuffle=True)
CV2 = model_selection.KFold(n_splits=K2,shuffle=True)

# Initialize error and complexity control
errors = np.zeros((K1))


n = 0
for train_index, test_index in CV.split(X):

    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    
    i = 0
    for train_index2, test_index2 in CV2.split(X_train):
        print('Crossvalidation fold: {0}/{1}'.format(n+1,i+1))    
        
        # extract training and test set for current CV fold
        X_train2 = X[train_index2,:]
        y_train2 = y[train_index2]
        X_test2 = X[test_index2,:]
        y_test2 = y[test_index2]

        i+=1

    # BASELINE
    baseline_guess = np.argmax(np.bincount(y_train))
    y_est = np.ones((1,y_test.shape[0]))*baseline_guess

    errors[n] = np.sum(y_est != y_test) / float(len(y_test))
    
    n+=1