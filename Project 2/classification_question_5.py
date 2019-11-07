from matplotlib.pyplot import figure, plot, xlabel, ylabel, show, subplot, semilogx, title, grid, legend, suptitle, tight_layout
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import mcnemar

# Gets data
from read_ecoli_data import *

# Mean op optimal lambda found from last exercise
lambda_opt = 0.06


K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True, random_state = 1)
for train_index, test_index in CV.split(X):
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]



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
s = 18
val = X_test[s:s+1,:]
x = X_test[s,:]


a = logreg.coef_ @ x
b = y_test[s]

# Apply softmax
c = np.exp(a)/np.sum(np.exp(a))

for i in range(len(classNames)):
    print(np.round(c[i],2))
print('Guess: ',np.argmax(c))  
print('Predicted: ', logreg.predict(val))
print('Real: ', b)  
    
scores = np.zeros(8) 
for i in range(len(classNames)):
    w  = logreg.coef_[i,:]
    a = w.dot(x)
    scores[i] = a
#    c = np.exp(a)/np.sum(np.exp(a))
#    print(np.max(c))
#scores -= np.max(scores)
print('Guess: ',np.argmax(np.exp(scores)/np.sum(np.exp(scores))))



