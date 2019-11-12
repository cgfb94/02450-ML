from matplotlib.pyplot import figure, plot, xlabel, ylabel, show, subplot, semilogx, title, grid, legend, suptitle, tight_layout
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from toolbox_02450 import mcnemar

# Gets data
from read_ecoli_data import *

# Standardizes data matrix so each column has mean 0 and std 1
X = (X - np.ones((N,1))*X.mean(0))/X.std(0)

# Mean op optimal lambda found from last exercise
lambda_opt = 1.29

#K = 10
#CV = model_selection.KFold(n_splits=K,shuffle=True, random_state = 1)
#for train_index, test_index in CV.split(X):
#    # extract training and test set for current CV fold
#    X_train = X[train_index,:]
#    y_train = y[train_index]
#    X_test = X[test_index,:]
#    y_test = y[test_index]

K = 10
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)



logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial',max_iter = 100000, tol=1e-4, C=1/lambda_opt, random_state = 1)
logreg.fit(X_train,y_train)

y_est = logreg.predict(X_test).T
error = np.sum(y_est != y_test) / len(y_test) 

# Look at coefficients
print('Coefficients:')

for i in range(len(classNames)):
    print(classNames[i],': ',np.round(logreg.coef_[i,:],2))


#%%
# Select test value

#index = 5

#x = X_test[index,:]
x = X_test

# Weights
w = logreg.coef_

# multiply x with weights
temp = logreg.coef_ @ x.T

# Compute the softmax
e = np.exp(temp)
theta = e / np.sum(e,0) # 

# Make prediction
myGuess = np.argmax(theta,0)

#prediction = logreg.predict(x.reshape(1,-1))[0]
prediction = logreg.predict(x)

real_value = y_test[:]

print('\nManual check for model predictions:\n')
print('My guess: ',myGuess,'\nPrediction: ',prediction,'\nreal value: ',real_value)