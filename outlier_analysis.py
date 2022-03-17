# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:32:42 2022

@author: bhupendra.singh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_analysis import *

def plot_distribution(d_0, d_1, d_2):
    plt.figure(figsize=(16,5))
    plt.subplot(1,3,1)
    sns.distplot(d_0)
    plt.subplot(1,3,2)
    sns.distplot(d_1)
    plt.subplot(1,3,3)
    sns.distplot(d_2)
    plt.show()
    
def remove_outlier(data):
    percentile25 = data.quantile(0.25)
    percentile75 = data.quantile(0.75)
    iqr = percentile75 - percentile25
    #upper_limit = percentile75 + 1.5 * iqr
    upper_limit = percentile75 + iqr
    #lower_limit = percentile25 - 1.5 * iqr
    lower_limit = percentile25 - iqr
    
    new_df = data[data < upper_limit]
    new_df = new_df[new_df > lower_limit]
    return new_df

def remove_outlier_by_value(dataset, outlier_column_name, cutoff_mx, cutoff_mi):
    
    d_selected = dataset[[outlier_column_name, 'SampleNo']]
    d_selected_outlier_removed = d_selected[d_selected[outlier_column_name] < cutoff_mx]
    d_selected_outlier_removed = d_selected_outlier_removed[d_selected_outlier_removed[outlier_column_name] > cutoff_mi]
    a = d_selected['SampleNo']
    b = d_selected_outlier_removed['SampleNo']
    outliers_indices = np.setdiff1d(a,b)
    return d_selected_outlier_removed, outliers_indices
    
dataset = pd.read_csv('dataset_pca.csv')

X = dataset.iloc[:,6]
Y = dataset.iloc[:,3]

d_0 = X[Y==0]
d_1 = X[Y==1]
d_2 = X[Y==2]

d_0_dataset = dataset[dataset['TrueClass'] == 0]
outlier_feature_name = 'pca_feature_1'
d_0_cleaned,b = remove_outlier_by_value(d_0_dataset, outlier_feature_name, 7.5, d_0_dataset[outlier_feature_name].min())

#plot_distribution(d_0,d_1,d_2)
perform_t_test_3_group(d_0, d_1, d_2)
print("######## perfroming kruskal wallis test ###########")
perform_kruskal_test_repeated(d_0,d_1,d_2)
print("after cleaning")
perform_t_test_3_group(d_0_cleaned[outlier_feature_name], d_1, d_2)
#plot_distribution(d_0_cleaned, d_1_cleaned, d_2_cleaned)

print("######## perfroming kruskal wallis test ###########")
perform_kruskal_test_repeated(d_0_cleaned[outlier_feature_name],d_1,d_2)

