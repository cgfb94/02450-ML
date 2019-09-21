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


#%%
data= pd.read_csv(r"C:\Users\callu\Desktop\ecoli.data", sep='\s+', header=None)
#%%
X = data.iloc[:,1:8].values
X = StandardScaler().fit_transform(X)
y = data.iloc[:,8:]
y.rename(columns={8:'Target'}, inplace=True)


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
print(myset)

#%%
sns.lmplot('PC1','PC2',final_df,hue='Target',fit_reg=False)


#%%


colors = range(len(labels))

xs = final_df['PC1']
ys = final_df['PC2']
zs = final_df['PC3']

#ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w', c=)#, colors=colors)



for ii in range(10):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10., azim=ii)
    for label in labels:
        #add data points 
        ax.scatter(xs=final_df.loc[final_df['Target']==label, 'PC1'], 
                    ys=final_df.loc[final_df['Target']==label,'PC2'], 
                    zs=final_df.loc[final_df['Target']==label,'PC3'],
                    alpha=0.5)
    ax.legend(labels, loc='best')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.savefig("hellofindme.png")
    plt.show()

#%%

#%%
