# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 11:28:25 2022

@author: bhupendra.singh
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

#dataset = pd.read_csv ('master_dataset_cleaned.csv')
dataset = pd.read_csv ('master_dataset.csv')
dataset['TrueClass'] = dataset['TrueClass'].astype(int)
#dataset = pd.read_csv ('master_dataset_GeMAPS.csv')

Y = dataset.iloc[:,3]
X = dataset.iloc[:,4:]

X = StandardScaler().fit_transform(X)

pca = PCA(0.95)

PC = pca.fit_transform(X)
PC_analysis = pca.explained_variance_ratio_
PC = pd.DataFrame(PC)

weighted_PCA_feature = np.zeros(X.shape[0])

i = 0
while i < X.shape[0]:
    j = 0
    sum_ = 0
    weighted_sum = 0
    while j < PC.shape[1]:
        weighted_sum += abs(PC.iloc[i,j]) * PC_analysis[j]
        sum_ += PC_analysis[j]
        j += 1
    weighted_PCA_feature[i] = weighted_sum/sum_
    i += 1

X = pd.DataFrame(X)

column_names = list()
column_names.append('SampleNo')
column_names.append('Subject')
column_names.append('UtteranceNo')
column_names.append('TrueClass')
column_names.append('weighted_PCA_feature')

i = 1
while i <= PC.shape[1]:
    column_names.append('pca_component_' + str(i)) 
    i += 1
pca_based_features = pd.DataFrame(dataset.iloc[:,:68], columns = column_names)
#pca_based_features = dataset.iloc[:,:3]
row = 0
while row < PC.shape[0]:
    col = 0
    while col < PC.shape[1]:
        pca_based_features.iloc[row, 5+col] = PC.iloc[row,col]
        col += 1
    row += 1
    
pca_based_features['weighted_PCA_feature'] = weighted_PCA_feature
pca_based_features['TrueClass'] = Y

pca_based_features['mean_PCA'] = 0
mn = 0
i = 0
while i < pca_based_features.shape[0]:
    mn = pca_based_features.iloc[i, 5:].mean()
    pca_based_features.iloc[i, 68] = mn
    i += 1

#data_0 = pca_based_features[pca_based_features['TrueClass'] == 0]
#data_1 = pca_based_features[pca_based_features['TrueClass'] == 1]
#data_2 = pca_based_features[pca_based_features['TrueClass'] == 2]
#
#d_0 = data_0['weighted_PCA_feature']
#d_1 = data_1['weighted_PCA_feature']
#d_2 = data_2['weighted_PCA_feature']
#
#plt.scatter(np.arange(1,d_0.shape[0]+1,1), d_0, color='red')
#plt.scatter(np.arange(1,d_1.shape[0]+1,1), d_1, color='green')
#plt.scatter(np.arange(1,d_2.shape[0]+1,1), d_2, color='blue')

pca_based_features.to_csv('./dataset_pca.csv', index=False)