
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

from sklearn import preprocessing

minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

X_0 = minmax_scale.fit_transform(X_0)
X_1 = minmax_scale.fit_transform(X_1)
X_2 = minmax_scale.fit_transform(X_2)

# print(pd.DataFrame(X_0.flatten()).describe())
# print(pd.DataFrame(X_1.flatten()).describe())
# print(pd.DataFrame(X_2.flatten()).describe())

from scipy.stats import f_oneway

#perform one-way ANOVA
F, p = f_oneway(X_0, X_1, X_2)
# print(F)
# print(p)
# print(ALL)

# import seaborn as sns

# sns.displot(ALL)
# plt.show()


