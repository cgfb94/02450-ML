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
lambda_opt = 0.044

# Split into test and training set
index = 303

X_train = X[0:index,:]
y_train = y[0:index]
X_test = X[index-1:-1,:]
y_test = y[index-1:-1]




logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial',max_iter = 100000, tol=1e-4, C=1/lambda_opt, random_state = 1)
logreg.fit(X_train,y_train)

y_est = logreg.predict(X_test).T
error = np.sum(y_est != y_test) / len(y_test) 

# Look at coefficients
print('Coefficients:')

for i in range(len(classNames)):
    print(classNames[i],': ',logreg.coef_[i,:])

