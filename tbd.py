# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 19:59:27 2022

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

def normalize_dataset(input_dataset):
    df_min_max_scaled = input_dataset.copy()  
    # apply normalization techniques
    for column in df_min_max_scaled.columns:
        df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    
    return df_min_max_scaled
#dataset = pd.read_csv ('master_dataset_cleaned.csv')
dataset = pd.read_csv ('master_dataset.csv')
dataset['TrueClass'] = dataset['TrueClass'].astype(int)

dataset.iloc[:,4:] = normalize_dataset(dataset.iloc[:,4:])

dataset['mean_feature'] = 0
i = 0
mn = 0
while i < dataset.shape[0]:
    dataset.iloc[i, 132] = dataset.iloc[i,4:132].mean()
    i += 1

dataset.to_csv('./tmp_dataset.csv', index=False)

