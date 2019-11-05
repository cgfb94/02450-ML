from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot
from sklearn import model_selection

# THIS CODE IS NOT RELEVANT FOR PROJET
# IT IS MAINLY FOR TESTING IDEAS
# THIS CODE IS NOT RELEVANT FOR PROJET
# IT IS MAINLY FOR TESTING IDEAS
# THIS CODE IS NOT RELEVANT FOR PROJET
# IT IS MAINLY FOR TESTING IDEAS

# Gets data
from read_ecoli_data import *


# Initialize training parameters
lambda_interval = np.logspace(-8, 2, 50)
lambda_interval = np.power(10.,range(-4,3))
L = len(lambda_interval)


# Set cross-validation parameters
K1 = 10
K2 = K1
CV = model_selection.KFold(n_splits=K1,shuffle=True)
CV2 = model_selection.KFold(n_splits=K2,shuffle=True)

# Initialize error and complexity control
train_error_rate2 = np.zeros((K2,L))
test_error_rate2 = np.zeros((K2,len(lambda_interval)))

train_error_rate = np.zeros((K1))
test_error_rate = np.zeros((K1))


min_error = np.zeros(K1)
s = np.zeros(K1)
opt_lambda = np.zeros(K1)


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
   
        
        for l in range(0,L):            
            logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial',max_iter = 100000, tol=1e-4, C=1/lambda_interval[l])
            logreg.fit(X_train2,y_train2)

            y_train_est2 = logreg.predict(X_train2).T
            y_test_est2 = logreg.predict(X_test2).T
            
            train_error_rate2[i,l] = np.sum(y_train_est2 != y_train2) / len(y_train2)
            test_error_rate2[i,l] = np.sum(y_test_est2 != y_test2) / len(y_test2)
        i+=1
        
    min_error[n] = np.min(test_error_rate2.mean(0))
    minArg = np.argmin(test_error_rate2.mean(0))
    s[n] = minArg+1
    opt_lambda[n] = lambda_interval[minArg]
    
    
    # Compute logistical regression with best lambda from inner fold
    logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial',max_iter = 100000, tol=1e-4, C=1/opt_lambda[n])
    logreg.fit(X_train,y_train)

    y_train_est = logreg.predict(X_train).T
    y_test_est = logreg.predict(X_test).T
    
    train_error_rate[n] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[n] = np.sum(y_test_est != y_test) / len(y_test)
    
    n+=1


### Plot for last inner fold
f = figure();
plt.semilogx(lambda_interval, train_error_rate2.mean(0)*100)
plt.semilogx(lambda_interval, test_error_rate2.mean(0)*100)

#plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error[-1]*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda[-1]),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error'],loc='upper right')
#plt.ylim([0, 4])
plt.grid()
plt.show()