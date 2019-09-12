import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd

# Data link:
# https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center

# Read data
filename = 'transfusion.data'
df = pd.read_csv(filename)
raw_data = df.get_values()

cols = range(df.shape[1]-1) 

X = raw_data[:, cols]

attributeNames = np.asarray(df.columns[cols])
classNames = ['Donater','Non donater']
C = len(classNames)

y = raw_data[:,-1] 

N, M = X.shape
# %% PCA
# Subtract mean value from data
Y = (X - np.ones((N,1))*X.mean(axis=0))/X.std(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)
V = V.T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Project the centered data onto principal component space
Z = Y @ V

# %% Plots
# Data attributes to be plotted
i = 1
j = 2

f = plt.figure()
plt.title('NanoNose data')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)

plt.legend(classNames)
plt.xlabel(attributeNames[i])
plt.ylabel(attributeNames[j])
plt.show()

# Plot variance explained
threshold = 0.9
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

# Plot PCA of the data
f = plt.figure()
plt.title('Transfusion data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.show()

# Bar plot of PCA
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Transfusion: PCA Component Coefficients')
plt.show()

