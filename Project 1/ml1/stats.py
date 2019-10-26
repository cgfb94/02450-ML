# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition.pca import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn')
color_map = ["b","g","lime","r","c","m","y","k"]
markers = ['o','s','s','s','s','.','.','v']

#%%
cols = ["Name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "Target"]

data= pd.read_csv(r"C:\Users\callu\Desktop\ecoli.data", sep='\s+', header=None, names=cols)
#%%
X = data.iloc[:,1:8].values
X = StandardScaler().fit_transform(X)
y = data.iloc[:,8:]
#%%
data.groupby('Target')['Name'].nunique().plot(kind='bar')
plt.show()
#%%
from pandas.plotting import scatter_matrix
sns.pairplot(data.drop(['lip', 'chg'], axis=1), hue="Target", palette=color_map, markers=markers, diag_kind='hist', size=2, kind='reg')
#scatter_matrix(data.drop(['lip', 'chg'], axis=1), alpha=0.2, figsize=(6, 6), diagonal='kde', hue='Target')
#%%
model = PCA(n_components=3)
principle_comps = model.fit_transform(X)
#%%
principle_df = pd.DataFrame(data=principle_comps, columns=["PC1","PC2","PC3"])
final_df = pd.concat([principle_df, y], axis=1)

#%%
new_y = y.values.tolist()
new_y = [entry[0] for entry in new_y]
#%%
labels = set(new_y)
print(labels)

#%%
sns.lmplot('PC1','PC2',final_df,hue='Target',fit_reg=False,size=10,palette=color_map, markers=markers)


#%%


colors = range(len(labels))

xs = final_df['PC1']
ys = final_df['PC2']
zs = final_df['PC3']


#%%
