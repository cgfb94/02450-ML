from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid, plot)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from sklearn.preprocessing import OneHotEncoder

from read_ecoli_data import *

# one hot encoding of cell type
ohe = OneHotEncoder(sparse=False,categories='auto')
x = ohe.fit_transform(y.reshape(-1, 1))

# Select the feature 'alm2' as new predictor and remove from data set
y = X[:,-1]
X = X[:,0:-1]

# Add one hot encoded to data set
X = np.concatenate((X,x),1)

# Standardizes data matrix so each column has mean 0 and std 1
X = (X - np.ones((N,1))*X.mean(0))/X.std(0)

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)

# Set attribute names and shape
attributeNames = attributeNames[0:-1]
attributeNames = [u'Offset']+attributeNames + [u'cp'] + [u'im'] + [u'imL'] + [u'imS'] + [u'imU'] + [u'om'] + [u'omL'] + [u'pp']
M = X.shape[1]

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)


# Values of lambda
lambdas = np.power(10.,range(-1,7))
#lambdas = np.arange(0.1,100,0.5)

# Initialize data
w = np.empty((M,K,len(lambdas)))
train_error = np.empty((K,len(lambdas)))
test_error = np.empty((K,len(lambdas)))
y = y.squeeze()


k = 0
for train_index, test_index in CV.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    
    # precompute terms
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    
    for l in range(0,len(lambdas)):
        # Compute parameters for current value of lambda and current CV fold
        lambdaI = lambdas[l] * np.eye(M)
        lambdaI[0,0] = 0 # remove bias regularization
        w[:,k,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Evaluate training and test performance
        train_error[k,l] = np.power(y_train-X_train @ w[:,k,l].T,2).mean(axis=0)
        test_error[k,l] = np.power(y_test-X_test @ w[:,k,l].T,2).mean(axis=0)

    k=k+1
    
minArg = np.argmin(np.mean(test_error,axis=0))

opt_val_err = np.min(np.mean(test_error,axis=0))
opt_lambda = lambdas[minArg]
train_err_vs_lambda = np.mean(train_error,axis=0)
test_err_vs_lambda = np.mean(test_error,axis=0)
mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))


f = figure()
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
semilogx(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()

# The difference from last plot here is that opt_lamda is not written as a power of 10
f = figure()
title('Optimal lambda: {0}'.format(opt_lambda))
semilogx(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()

f = figure()
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
#legend(attributeNames[1:], loc='best')


print('Weights for best regularization parameter:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(mean_w_vs_lambda[m,minArg],3)))
