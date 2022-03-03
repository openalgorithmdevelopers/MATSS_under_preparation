
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from scipy.stats import f_oneway
from scipy import stats

def set_limits(IQR, percentile75, percentile25):  
    upper_limit = percentile75
    lower_limit = percentile25
    # upper_limit = percentile75 + 1.5 * IQR
    # lower_limit = percentile25 - 1.5 * IQR
    return upper_limit, lower_limit

def findOutliers(df, column_name):
    percentile25 = df[column_name].quantile(0.25)
    percentile75 = df[column_name].quantile(0.75)
    IQR = percentile75 - percentile25

    upper_limit, lower_limit = set_limits(IQR, percentile75, percentile25)
    upper_outliers = df[df[column_name] > upper_limit]
    lower_outliers = df[df[column_name] < lower_limit]

    new_df = df[df[column_name] < upper_limit]
    new_df = df[df[column_name] > lower_limit]
    # print(lower_limit)

    # sns.boxplot(df[column_name])
    # sns.displot(df[column_name])
    # plt.show()

    # print(df.shape)
    # print(new_df.shape)
    return new_df

totalSubjects = 21
totalUtterances = 60    #the number of utterances of words in a folder for every subject
featureName = "mfcc"
FEATURE_SIZE = 128

dataset_all = pd.read_csv ('master_dataset.csv')
Y = dataset_all.iloc[:,3]
X = dataset_all.iloc[:,4:]

X_0 = dataset_all[dataset_all["TrueClass"] == 0]
X_0 = X_0.iloc[:,4:]
X_1 = dataset_all[dataset_all["TrueClass"] == 1]
X_1 = X_1.iloc[:,4:]
X_2 = dataset_all[dataset_all["TrueClass"] == 2]
X_2 = X_2.iloc[:,4:]



minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

X_0 = minmax_scale.fit_transform(X_0)
X_1 = minmax_scale.fit_transform(X_1)
X_2 = minmax_scale.fit_transform(X_2)

# print(X_0.shape[0])
X_0_summed = np.zeros(X_0.shape[0])
X_0_summed = np.mean(X_0, axis=1)
X_1_summed = np.zeros(X_1.shape[0])
X_1_summed = np.mean(X_1, axis=1)
X_2_summed = np.zeros(X_2.shape[0])
X_2_summed = np.mean(X_2, axis=1)

# print(pd.DataFrame(X_0.flatten()).describe())
# print(pd.DataFrame(X_1.flatten()).describe())
# print(pd.DataFrame(X_2.flatten()).describe())

# print(X_0_summed.shape)
# print(pd.DataFrame({'features_summed': X_0_summed}))
df_0 = pd.DataFrame({'features_summed': X_0_summed})
df_1 = pd.DataFrame({'features_summed': X_1_summed})
df_2 = pd.DataFrame({'features_summed': X_2_summed})
df_new_0 = findOutliers(df_0, 'features_summed')
df_new_1 = findOutliers(df_1, 'features_summed')
df_new_2 = findOutliers(df_2, 'features_summed')


# plt.figure(figsize=(16,5))
# plt.subplot(1,2,1)
# sns.distplot(df_0)
# plt.subplot(1,2,2)
# sns.distplot(df_new_0)
# plt.show()


# #perform one-way ANOVA
# F, p = f_oneway(df_0, df_1, df_2)
# print(F)
# print(p)
# # print(ALL)

# t_value,p_value=stats.ttest_ind(df_0,df_1)
t_value,p_value=stats.ttest_ind(df_new_0,df_new_1)
print(t_value)
print(p_value)

#sns.boxplot(X_0.flatten())
# sns.displot(X_0_summed)
# plt.show()


