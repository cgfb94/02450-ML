from matplotlib.pyplot import figure, plot, xlabel, ylabel, show, subplot, semilogx, title, grid, legend, suptitle, tight_layout
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import mcnemar

# Gets data
from read_ecoli_data import *

# Set cross-validation parameters
K1 = 10
K2 = K1
CV = model_selection.KFold(n_splits=K1,shuffle=True, random_state = 1)
CV2 = model_selection.KFold(n_splits=K2,shuffle=True, random_state =1)


# Initialize error and complexity control - LOGREG
lambda_interval = np.logspace(-3, 1, 10)
#lambda_interval = np.power(10.,range(-4,3))
L_LOG = len(lambda_interval)

error2_LOG = np.zeros((K2,len(lambda_interval)))
error_LOG = np.zeros((K1))
min_error_LOG = np.zeros(K1)
s_LOG = np.zeros(K1)
opt_lambda = np.zeros(K1)


# Initialize error and complexity control - KNN
L = 40  # Maximum number of neighbors for KNN
L_list = np.arange(1,L+1,1)
errors_KNN = np.zeros((K1))
errors2_KNN = np.zeros((K2,L))
s_KNN = np.zeros(K1)
x_KNN = [0] * K1

# Initialize error and complexity control - baseline
errors_baseline = np.zeros((K1))

yhat = [];
y_true = []
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
    
        # KNN
        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for l in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l);
            knclassifier.fit(X_train2, y_train2);
            y_est2 = knclassifier.predict(X_test2);
            errors2_KNN[i,l-1] = np.sum(y_est2 != y_test2) / float(len(y_est2))
            
        # LOGISTICAL REGRESSION   
        for l in range(0,L_LOG):            
            logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial',max_iter = 100000, tol=1e-4, C=1/lambda_interval[l], random_state = 1)
            logreg.fit(X_train2,y_train2)

            y_test_est2 = logreg.predict(X_test2).T

            error2_LOG[i,l] = np.sum(y_test_est2 != y_test2) / len(y_test2)
        i+=1
    
    
    # KNN
    # Find which element corresponds to the smallest error
    minArg = np.argmin(errors2_KNN.mean(0))
    s_KNN[n] = minArg+1
    x_KNN[n] = L_list[minArg]
    
    # Compute the best KNN from the inner fold
    knclassifier = KNeighborsClassifier(n_neighbors=x_KNN[n]);
    knclassifier.fit(X_train, y_train);
    y_est_KNN = knclassifier.predict(X_test);
    errors_KNN[n] = np.sum(y_est_KNN != y_test) / float(len(y_est_KNN))
    
    
    # LOGISTICAL REGRESSION
    min_error_LOG[n] = np.min(error2_LOG.mean(0))
    minArg = np.argmin(error2_LOG.mean(0))
    s_LOG[n] = minArg+1
    opt_lambda[n] = lambda_interval[minArg]
    
    # Compute logistical regression with best lambda from inner fold
    logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial',max_iter = 100000, tol=1e-4, C=1/opt_lambda[n], random_state = 1)
    logreg.fit(X_train,y_train)
    y_est_LOG = logreg.predict(X_test).T
    error_LOG[n] = np.sum(y_est_LOG != y_test) / len(y_test)    
    
    # BASELINE
    baseline_guess = np.argmax(np.bincount(y_train))
    y_est_BASE = np.ones((y_test.shape[0]), dtype = int)*baseline_guess
    errors_baseline[n] = np.sum(y_est_BASE != y_test) / float(len(y_test))
    
    # Combine all predictions in array
    dy = []
    dy.append(y_est_BASE)
    dy.append(y_est_KNN)
    dy.append(y_est_LOG)
    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    
    y_true.append(y_test)
    n+=1
   
    
# combine all predictions and real values
y_true = np.concatenate(y_true)
yhat = np.concatenate(yhat)


print('Errors KNN:\tErrors baseline\tErrors LOGREG')
for m in range(K1):   
    print(' ',np.round(errors_KNN[m],2),'\t\t',np.round(errors_baseline[m],2),'\t\t',np.round(error_LOG[m],2))
    
    
# PLOTS 
dpi = 75 # Sets dpi for plots
save_plots = False

# Plot the classification error rate for last inner fold for KNN
f = figure(dpi=dpi)

subplot(2, 1, 1)
plot(L_list,errors2_KNN.mean(0)*100,'-o')
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')

subplot(2, 1, 2)
semilogx(lambda_interval, error2_LOG.mean(0)*100,'-or')
xlabel('Regularization strength, $\log_{10}(\lambda)$')
ylabel('Classification error rate (%)')

tight_layout()
show()
f.savefig('./figures/inner_fold_classification.png', bbox_inches='tight') if save_plots else 0

#%% Statistical evaluation Setup I
    
alpha = 0.05


print('A : Baseline\nB : KNN')
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)
print('theta: ',np.round(thetahat,2),' CI: ',np.round(CI,2),' p: ',np.round(p,3))
print('\n')
print('A : Baseline\nB : Logistical Regression')
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,2], alpha=alpha)
print('theta: ',np.round(thetahat,2),' CI: ',np.round(CI,2),' p: ',np.round(p,3))
print('\n')
print('A : KNN\nB : Logistical Regression')
[thetahat, CI, p] = mcnemar(y_true, yhat[:,1], yhat[:,2], alpha=alpha)
print('theta: ',np.round(thetahat,2),' CI: ',np.round(CI,2),' p: ',np.round(p,3))


#print("\ntheta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)



