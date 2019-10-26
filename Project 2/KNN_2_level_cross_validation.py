from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

# Gets data
from read_ecoli_data import *

# Maximum number of neighbors
L=40
L_list = np.arange(1,L+1,1)


# Set cross-validation parameters
K1 = 10
K2 = K1
CV = model_selection.KFold(n_splits=K1,shuffle=True)
CV2 = model_selection.KFold(n_splits=K2,shuffle=True)

# Initialize error and complexity control
errors = np.zeros((K1))
errors2 = np.zeros((K2,L))
s = np.zeros(K1)


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
    
        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for l in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l);
            knclassifier.fit(X_train2, y_train2);
            y_est2 = knclassifier.predict(X_test2);
            errors2[i,l-1] = np.sum(y_est2 != y_test2) / float(len(y_est2))
        i+=1
    
    # Find which element corresponds to the smallest error
    minArg = np.argmin(errors2.mean(0))
    s[n] = minArg+1
    minNeighbors = L_list[minArg]
    
    # Compute the best KNN from the inner fold
    knclassifier = KNeighborsClassifier(n_neighbors=minNeighbors);
    knclassifier.fit(X_train, y_train);
    y_est = knclassifier.predict(X_test);
    errors[n] = np.sum(y_est != y_test) / float(len(y_est))
    
    n+=1


# THIS PLOT IS CURRENTLY WRONG
# THIS PLOT IS CURRENTLY WRONG
# THIS PLOT IS CURRENTLY WRONG

# Plot the classification error rate for last inner fold
figure()
plot(100*sum(errors2,0)/N)
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
show()