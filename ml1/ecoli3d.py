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
print(labels)

#%%
sns.lmplot('PC1','PC2',final_df,hue='Target',fit_reg=False)


#%%


colors = range(len(labels))

xs = final_df['PC1']
ys = final_df['PC2']
zs = final_df['PC3']

#ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w', c=)#, colors=colors)

color_map = ["b","g","lime","r","c","m","y","k"]
markers = ['o','s','s','s','s','.','.','v']



#%%
cols = ["Name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "Target"]
labels = ['cp' , 'im', 'imL', 'imS', 'imU', 'om', 'omL', 'pp']

for ii in range(1):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.patch.set_facecolor('white')
    ax.view_init(elev=10., azim=60)
    for label, colour, marker in zip(labels, color_map, markers):
        #add data points 
        ax.scatter(xs=final_df.loc[final_df['Target']==label, 'PC1'], 
                    ys=final_df.loc[final_df['Target']==label,'PC2'], 
                    zs=final_df.loc[final_df['Target']==label,'PC3'],
                    alpha=0.5, c=colour, marker=marker)
    ax.legend(labels, loc='best')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    #ax.set_xlim3d(0,max(Vr))
    ax.set_ylim3d(-2,4)
    ax.set_zlim3d(-2,4)
    plt.tight_layout()
    plt.savefig("hellofindme.png")
    plt.show()

#%%

#%%
