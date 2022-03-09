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

totalSubjects = 21
totalUtterances = 60    #the number of utterances of words in a folder for every subject
featureName = "mfcc"
FEATURE_SIZE = 128

def plotHistogram(data):
    fig, axs = plt.subplots(1, 1,
                            figsize =(10, 7),
                            tight_layout = True)
     
    axs.hist(data, bins = 40)
     
    # Show plot
    plt.show()

# each class should be equal to find equal indices
def plotScatter(data_0, data_1, data_2):
    indices = np.arange(1,len(data_2)+1, 1)
    plt.scatter(indices, data_0, color = "red")
    plt.scatter(indices, data_1, color = "green")
    plt.scatter(indices, data_2, color = "blue")
## #sns.boxplot(X_0.flatten())
#sns.displot(X_0_mean_feature)
#plt.show()

def rescale(data, mi, mx):
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(mi, mx))
    return(minmax_scale.fit_transform(data))

#rescaels data with mean = 0 and vriance = 1
def rescaleStandard(data):
    scaled = StandardScaler().fit_transform(data)
    return scaled

def perform_PCA(data):
    pca = PCA(n_components=4)
    principalComponents = pca.fit_transform(data)
    return principalComponents

def getMeanFeatureSet(data):
    data_mean_feature = np.mean(data, axis=1)
    return data_mean_feature

def perform_t_test(group1, group2):
    print(group1.shape[0])
    if(group1.shape[0] == group2.shape[0]):
        t_value,p_value=stats.ttest_rel(group1,group2)        
    else:
        t_value,p_value=stats.ttest_ind(group1,group2)
    
    print('Test statistic is %f'%float("{:.6f}".format(t_value)))
    
    print('p-value for two tailed test is %f'%p_value)

def clusterBasedClassification(X, Y):
    km = KMeans(3)
    
    
    clusters = km.fit_predict(X)
    clusters = pd.DataFrame(clusters)
    clusters.columns = ['cluster']
    C_0 = X[clusters['cluster'] == 0]
    C_1 = X[clusters['cluster'] == 1]
    C_2 = X[clusters['cluster'] == 2]
    
    
    PC = perform_PCA(X)
    PC = pd.DataFrame(PC)
    
    Y = pd.DataFrame(Y)
    Y.columns = ["TrueClass"]
    
    d_0 = PC[clusters['cluster'] == 0]
    d_1 = PC[clusters['cluster'] == 1]
    d_2 = PC[clusters['cluster'] == 2]
    return d_0, d_1, d_2


######################### main code starts here ###################################

dataset = pd.read_csv ('master_dataset_GeMAPS.csv')

Y = dataset.iloc[:,3]
X = dataset.iloc[:,4:]
X = rescaleStandard(X)

#########################
#trying with cluster to see what class
#the KNN finds 
d0, d1, d2 = clusterBasedClassification(X,Y)
result = pd.concat([d0,d1,d2])

rus = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
#X, Y = rus.fit_resample(X, Y)

result, Y = rus.fit_resample(result, Y)
result['target'] = Y
P0 = result[result['target'] == 0]
P1 = result[result['target'] == 1]
P2 = result[result['target'] == 2]

plotScatter(P0.iloc[:,0], P1.iloc[:,0], P2.iloc[:,0])
plt.show()
perform_t_test(P0.iloc[:,0], P0.iloc[:,1])

######################################
# perform the analysis with the original classes
Y = dataset.iloc[:,3]
X = dataset.iloc[:,4:]
X = rescaleStandard(X)

#X, Y = rus.fit_resample(X, Y)

PC = perform_PCA(X)
PC = pd.DataFrame(PC)

Y = pd.DataFrame(Y)
Y.columns = ["TrueClass"]

d_0 = PC[Y["TrueClass"] == 0]
d_1 = PC[Y["TrueClass"] == 1]
d_2 = PC[Y["TrueClass"] == 2]

#plotScatter(d_0.iloc[:,0], d_1.iloc[:,0], d_2.iloc[:,0])
print("Performing Welch t test btw 0 and 1")
perform_t_test(d_0[0], d_0[1])

print("Performing Welch t test btw 1 and 2")
perform_t_test(d_1[0], d_2[1])

print("Performing Welch t test btw 0 and 2")
perform_t_test(d_0[0], d_1[1])
#perform_t_test(d_0.iloc[:,0], d_1.iloc[:,0])

# from scipy.stats import f_oneway

# #perform one-way ANOVA
# F, p = f_oneway(X_0, X_1, X_2)
# # print(F)
# # print(p)
# # print(ALL)