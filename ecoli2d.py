# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition.pca import PCA
from sklearn.preprocessing import StandardScaler


#%%
data= pd.read_csv(r"C:\Users\callu\Desktop\ecoli.data", sep='\s+', header=None)
#%%
X = data.iloc[:,1:8].values
X = StandardScaler().fit_transform(X)
y = data.iloc[:,8:]
y.rename(columns={8:'Target'}, inplace=True)


model = PCA(n_components=2)
principle_comps = model.fit_transform(X)
#%%
principle_df = pd.DataFrame(data=principle_comps, columns=["PC1","PC2"])
final_df = pd.concat([principle_df, y], axis=1)

#%%
new_y = y.values.tolist()
new_y = [entry[0] for entry in new_y]
#%%
labels = set(new_y)
print(myset)

#%%
sns.lmplot('PC1','PC2',final_df,hue='Target',fit_reg=False)

#%%
model.explained_variance_ratio_

#%%
model.explained_variance_

#%%
model.get_covariance()

#%%
model.mean_

#%%
model.singular_values_

#%%
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = wines['residual sugar']
ys = wines['fixed acidity']
zs = wines['alcohol']
ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')

ax.set_xlabel('Residual Sugar')
ax.set_ylabel('Fixed Acidity')
ax.set_zlabel('Alcohol')

plt.show()