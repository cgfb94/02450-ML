import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
plt.style.use('seaborn')
from read_ecoli_data import *

# THIS IS CURRENTLY AN OUTDATED FILE - BUT MAY STILL HAVE USE
# THIS IS CURRENTLY AN OUTDATED FILE - BUT MAY STILL HAVE USE
# THIS IS CURRENTLY AN OUTDATED FILE - BUT MAY STILL HAVE USE
# THIS IS CURRENTLY AN OUTDATED FILE - BUT MAY STILL HAVE USE
# THIS IS CURRENTLY AN OUTDATED FILE - BUT MAY STILL HAVE USE

# %% Linear model
import sklearn.linear_model as lm

# Use all data but the last row for prediction - NOT NESSESARY but I wanted to try :) 
X_lm = X[0:-1,:]
y_lm = y[0:-1]


# Fit logistic regression model
model = lm.logistic.LogisticRegression()
model = model.fit(X_lm,y_lm)

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X_lm)

y_est_cp = model.predict_proba(X_lm)[:, 0]
y_est_im = model.predict_proba(X_lm)[:, 1]
y_est_imL = model.predict_proba(X_lm)[:, 2]
y_est_imS = model.predict_proba(X_lm)[:, 3]
y_est_imU = model.predict_proba(X_lm)[:, 4]
y_est_om = model.predict_proba(X_lm)[:, 5]
y_est_omL = model.predict_proba(X_lm)[:, 6] 
y_est_pp = model.predict_proba(X_lm)[:, 7]

# Define a new data object (new type of wine), as in exercise 5.1.7
x = X[-1,:].reshape(1,-1)
# Evaluate the probability of x being pp (class=7) 
x_class = model.predict_proba(x)[0,7]

# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y_lm) / float(len(y_est))

# Display classification results
print('\nProbability of given sample being pp: {0:.4f}'.format(x_class))
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = plt.figure();
class0_ids = np.nonzero(y_lm==0)[0].tolist()
plt.plot(class0_ids, y_est_cp[class0_ids], '.y')
class1_ids = np.nonzero(y_lm==1)[0].tolist()
plt.plot(class1_ids, y_est_cp[class1_ids], '.r')
plt.xlabel('Data object (Protein Location)'); plt.ylabel('Predicted prob. of class cp');
plt.legend(['cp', 'im'])
plt.ylim(-0.01,1.5)
plt.show()


estimation = y_est_cp

f = plt.figure();
for i in range(C):
    class_ids = np.nonzero(y_lm == i)[0].tolist()
    plt.plot(class_ids,estimation[class_ids],'.')
plt.xlabel('Data object (Protein Location)'); plt.ylabel('Predicted prob. of class cp');
plt.legend(classNames)
plt.ylim(-0.01,1.5)
plt.show()    