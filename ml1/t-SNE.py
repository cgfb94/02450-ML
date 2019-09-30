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
scatter_matrix(data, alpha=0.2, figsize=(12, 12), diagonal='kde')
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
sns.lmplot('PC1','PC2',final_df,hue='Target',fit_reg=False,size=10)


#%%


colors = range(len(labels))

xs = final_df['PC1']
ys = final_df['PC2']
zs = final_df['PC3']

#%%
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(X)

final_df['tsne-2d-one'] = tsne_results[:,0]
final_df['tsne-2d-two'] = tsne_results[:,1]
#%%
test1 = ['o','s','s','s','s','.','.','v']
#plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="Target",
    data=final_df,
    legend="full",
    palette=color_map, markers= test1,
    alpha=0.5
)

#%%
from sklearn.manifold import TSNE
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(X)

final_df['tsne-3d-one'] = tsne_results[:,0]
final_df['tsne-3d-two'] = tsne_results[:,1]
final_df['tsne-3d-three'] = tsne_results[:,2]
#%%
xs = final_df['tsne-3d-one']
ys = final_df['tsne-3d-two']
zs = final_df['tsne-3d-three']

for ii in range(1):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.patch.set_facecolor('white')
    ax.view_init(elev=10., azim=45)
    for label in labels:
        #add data points 
        ax.scatter(xs=final_df.loc[final_df['Target']==label, 'tsne-3d-one'], 
                    ys=final_df.loc[final_df['Target']==label,'tsne-3d-two'], 
                    zs=final_df.loc[final_df['Target']==label,'tsne-3d-three'],
                    alpha=0.5)
    ax.legend(labels, loc='best')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    plt.savefig("hellofindme.png")
    plt.show()

#%%
