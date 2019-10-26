from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot
from scipy.io import loadmat
from sklearn import model_selection, tree
import numpy as np
from read_ecoli_data import *

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)


# K-fold crossvalidation
K1 = 10
K2 = K1
CV = model_selection.KFold(n_splits=K1,shuffle=True)
CV2 = model_selection.KFold(n_splits=K2,shuffle=True)


# Initialize variable
Error_train = np.ones(K1)
Error_test = np.ones(K1)
Error_train2 = np.ones((K2,len(tc)))
Error_test2 = np.ones((K2,len(tc)))

k=0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}...'.format(k+1,K1))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]
    
    
    # 2nd cross-validation level
    j = 0
    for train_index2, test_index2 in CV2.split(X_train):
        X_train2, y_train2 = X_train[train_index2,:], y[train_index2]
        X_test2, y_test2 = X[test_index2,:], y[test_index2]
        
        # Find the best tree depth
        for i in range(len(tc)):
            t = tc[i]            
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc = dtc.fit(X_train2,y_train2.ravel())
            y_est_test2 = dtc.predict(X_test2)
            y_est_train2 = dtc.predict(X_train2)
            # Evaluate misclassification rate over train/test data (in this CV fold)
            misclass_rate_test2 = np.sum(y_est_test2 != y_test2) / float(len(y_est_test2))
            misclass_rate_train2 = np.sum(y_est_train2 != y_train2) / float(len(y_est_train2))
            Error_test2[j,i], Error_train2[j,i] = misclass_rate_test2, misclass_rate_train2
        j+=1
            
    # Select optimal model for this cross validation
#    s = np.argmin(Error_test2[:,k])
    s = np.argmin(Error_test2.mean(0))
    minTreeDebth = tc[s]
    print('\nOptimal tree debth: {0}\n'.format(minTreeDebth))
    
    # Train best model from inner cross validation        
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=minTreeDebth)
    dtc = dtc.fit(X_train,y_train.ravel())
    y_est_test = dtc.predict(X_test)
    y_est_train = dtc.predict(X_train)
        
    misclass_rate_test = np.sum(y_est_test != y_test) / float(len(y_est_test))
    misclass_rate_train = np.sum(y_est_train != y_train) / float(len(y_est_train))
    Error_test[k], Error_train[k] = misclass_rate_test, misclass_rate_train
    k+=1

    
# Plot the last fold 2 model complexity
f = figure()
boxplot(Error_test2)
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K2))

  
# Plot of the errors for each outher fold
f = figure()
plot(np.arange(0,K1,1), Error_train)
plot(np.arange(0,K1,1), Error_test)
xlabel('Outer fold')
ylabel('Error (misclassification rate, CV K={0})'.format(K1))
legend(['Error_train','Error_test'])
    
show()
#
#minTreeDebth = tc[Error_test.mean(1).argmin()]
#print('\nOptimal tree debth: {0}\n'.format(minTreeDebth))
