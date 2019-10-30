import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
plt.style.use('seaborn')

# This file reads the ecoli data so the others don't have to

# Data link:
# https://archive.ics.uci.edu/ml/datasets/ecoli

filename = '../ecoli.csv'
# Data fields
fields = ['mcg', 'gvh','lip','chg','aac','alm1','alm2']
df = pd.read_csv(filename)
raw_data = df.get_values()
df2 = pd.read_csv(filename,usecols=fields)
raw_data2 = df2.get_values()

cols = range(1,8) 

# Read data:
X = raw_data2[:, :]
sequence_name = raw_data[:,0] # Store names
classLabels = raw_data[:,-1] # Class location site 

attributeNames = np.asarray(df.columns[cols])
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.array([classDict[cl] for cl in classLabels])

C = len(classNames)

N, M = X.shape

# Make attribute names string array
strs = ["" for x in range(M)]
for i in range(M):
    strs[i] = attributeNames[i]
attributeNames = strs