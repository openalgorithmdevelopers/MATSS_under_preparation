# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:11:05 2022

@author: bhupendra.singh
"""
import os

from data_analysis import *

def normalize_dataset(input_dataset):
    df_min_max_scaled = input_dataset.copy()  
    # apply normalization techniques
    for column in df_min_max_scaled.columns:
        df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    
    return df_min_max_scaled


sns.set(rc={"figure.dpi":400, 'savefig.dpi':400})


dataset_PCA = pd.read_csv('dataset_pca.csv')
dataset_PCA['TrueClass'] = dataset_PCA['TrueClass'].astype(int)
dataset_Master = pd.read_csv('master_dataset.csv')

#os.system("generate_PCA_dataset.py 1")

component = 'pca_component_1'
d_0 = dataset_PCA[dataset_PCA['TrueClass'] == 0]
d_1 = dataset_PCA[dataset_PCA['TrueClass'] == 1]
d_2 = dataset_PCA[dataset_PCA['TrueClass'] == 2]
#plotScatter(d_0[component], d_1[component], d_2[component])

######## Scatter plot ####################
fig, axs = plt.subplots(nrows=2)

ax1 = sns.scatterplot(data=dataset_PCA, x="SampleNo", y="pca_component_1", hue="TrueClass", palette="deep", ax=axs[0])
ax2 = sns.scatterplot(data=dataset_PCA, x="SampleNo", y="pca_component_2", hue="TrueClass", palette="deep", ax=axs[1])
ax1.set(xlabel='Sample No.',
       ylabel='Principal Component 1')
ax2.set(xlabel='Sample No.',
       ylabel='Principal Component 2')
#fig = pt.get_figure()
fig.savefig('pca_scatter.png')

####### Box plot ####################
pt, axs = plt.subplots(nrows=2)
ax3 = sns.boxplot(x="TrueClass", y='pca_component_1', data=dataset_PCA,  ax=axs[0])
ax4 = sns.boxplot(x="TrueClass", y='pca_component_2', data=dataset_PCA,  ax=axs[1])
ax3.set(xlabel='True Class',
       ylabel='Principal Component 1')
ax4.set(xlabel='True Class',
       ylabel='Principal Component 2')
pt.savefig('pca_box.png')