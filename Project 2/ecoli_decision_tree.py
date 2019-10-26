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



# %% Make decision tree
from sklearn import tree
from os import getcwd
from platform import system
from toolbox_02450 import windows_graphviz_call
from matplotlib.image import imread

# Fit regression tree classifier, Gini split criterion, no pruning
criterion='gini'
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=15)

# Use all data but the last row
X_tree = X[0:-1,:]
y_tree = y[0:-1]

dtc = dtc.fit(X_tree,y_tree)

fname='tree_' + criterion + '_ecoli_data'
# Export tree graph .gvz file to parse to graphviz
out = tree.export_graphviz(dtc, out_file=fname + '.gvz', feature_names=attributeNames)
    
# Needs a png to 
#if system() == 'Windows':
#    # N.B.: you have to update the path_to_graphviz to reflect the position you 
#    # unzipped the software in!
#    path_to_graphviz = r'C:\Program Files (x86)\release'
#    windows_graphviz_call(fname=fname,
#                          cur_dir=getcwd(),
#                          path_to_graphviz=path_to_graphviz)
#    plt.figure(figsize=(12,12))
#    plt.imshow(imread(fname + '.png'))
#    plt.box('off'); plt.axis('off')
#    plt.show()

# %% Make a prediction

# Define a new data object (new type of wine) with the attributes given in the text
#x = np.array([0.38, 0.4 , 0.48, 0.5 , 0.63, 0.25, 0.35]).reshape(1,-1)

# Test the last row
x = X[-1,:].reshape(1,-1)

# Evaluate the classification tree for the new data object
x_class = dtc.predict(x)[0]

# Print results
print('\nNew object attributes:')
for i in range(len(attributeNames)):
    print('{0}: {1}'.format(attributeNames[i],x[0][i]))
print('\nClassification result:')
print(classNames[x_class])
print('\nReal answer:')
print(classNames[y[-1]])

