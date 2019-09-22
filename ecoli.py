import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
plt.style.use('seaborn')

# Data link:
# https://archive.ics.uci.edu/ml/datasets/ecoli

filename = 'ecoli.csv'
# Data fields
fields = ['mcg', 'gvh','lip','chg','aac','alm1','alm2']
df = pd.read_csv(filename)
raw_data = df.get_values()
df2 = pd.read_csv(filename,usecols=fields)
raw_data2 = df2.get_values()

cols = range(1,8) 

# Read data:
X = raw_data2[:, :]
sequence_name = raw_data[:,0] # Don't know if this attribute is useful at all
classLabels = raw_data[:,-1] # Class location site 


attributeNames = np.asarray(df.columns[cols])
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.array([classDict[cl] for cl in classLabels])

C = len(classNames)

N, M = X.shape
# %% PCA

# Make 1-of-K-encoding
#onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
#y_reshaped = y.reshape(len(y), 1)
#onehot_encoded = onehot_encoder.fit_transform(y_reshaped)
#X = np.concatenate((X,onehot_encoded),1)
#N, M = X.shape

# Subtract mean value from data
Y = (X - np.ones((N,1))*X.mean(axis=0))

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)
V = V.T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Project the centered data onto principal component space
Z = Y @ V
# %% Correlation
f1 = plt.figure(dpi=300)
corr = df2.corr()
ax = sns.heatmap(corr,vmin=-1, vmax=1, center=0,square=True)
plt.title('Variable Correlation')
f1.savefig('./figures/correlation.png', bbox_inches='tight')
plt.show()
    

# %% Plots
# Data attributes to be plotted
i = 0
j = 1

f2 = plt.figure()
plt.title('Ecoli data')

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
f3 = plt.figure()
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
i = 0
j = 1

f4 = plt.figure(dpi=300)
plt.title('Ecoli data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
f4.savefig('./figures/PC1_PC2_plot.png', bbox_inches='tight')
plt.show()

# Bar plot of PCA
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
f5 = plt.figure(dpi = 300)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Ecoli: PCA Component Coefficients')
f5.savefig('./figures/PCA_bar_plot.png', bbox_inches='tight')
plt.show()