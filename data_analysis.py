
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

def plotHistogram(data):
    fig, axs = plt.subplots(1, 1,
                            figsize =(10, 7),
                            tight_layout = True)
     
    axs.hist(data, bins = 40)
     
    # Show plot
    plt.show()


## #sns.boxplot(X_0.flatten())
#sns.displot(X_0_mean_feature)
#plt.show()


def perform_t_test(group1, group2):
    t_value,p_value=stats.ttest_rel(group1,group2)
    
    print('Test statistic is %f'%float("{:.6f}".format(t_value)))
    
    print('p-value for two tailed test is %f'%p_value)

dataset = pd.read_csv ('master_dataset.csv')
Y = dataset.iloc[:,3]
X = dataset.iloc[:,4:]

X_0 = dataset[dataset["TrueClass"] == 0]
X_0 = X_0.iloc[:,4:]
X_1 = dataset[dataset["TrueClass"] == 1]
X_1 = X_1.iloc[:,4:]
X_2 = dataset[dataset["TrueClass"] == 2]
X_2 = X_2.iloc[:,4:]


def rescale(data, mi, mx):
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(mi, mx))
    return(minmax_scale.fit_transform(data))

X_0 = rescale(X_0, 0, 1)
X_1 = rescale(X_1, 0, 1)
X_2 = rescale(X_2, 0, 1)

X_0_mean_feature = np.mean(X_0, axis=1)
X_1_mean_feature = np.mean(X_1, axis=1)
X_2_mean_feature = np.mean(X_2, axis=1)

perform_t_test(X_0_mean_feature[:150], X_1_mean_feature[:150])


#print(pd.DataFrame(X_0.flatten()).describe())
#print(pd.DataFrame(X_1.flatten()).describe())
#print(pd.DataFrame(X_2.flatten()).describe())

# from scipy.stats import f_oneway

# #perform one-way ANOVA
# F, p = f_oneway(X_0, X_1, X_2)
# # print(F)
# # print(p)
# # print(ALL)

km = KMeans(n_clusters=3)

# Creating histogram