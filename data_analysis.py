import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg
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
def plotScatter(d_0,d_1,d_2):
    plt.scatter(np.arange(1,d_0.shape[0]+1,1), d_0, color='red')
    plt.scatter(np.arange(1,d_1.shape[0]+1,1), d_1, color='green')
    plt.scatter(np.arange(1,d_2.shape[0]+1,1), d_2, color='blue')
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
    #pca = PCA(n_components=2)
    pca = PCA(0.95)
    pca = PCA(n_components=4)
    principalComponents = pca.fit_transform(data)
    return principalComponents, pca.explained_variance_ratio_

def getMeanFeatureSet(data):
    data_mean_feature = np.mean(data, axis=1)
    return data_mean_feature

def perform_t_test(group1, group2):
    if(group1.shape[0] == group2.shape[0]):
        t_value,p_value=stats.ttest_rel(group1,group2)        
    else:
        t_value,p_value=stats.ttest_ind(group1,group2)
    
    print('Test statistic is %f'%float("{:.6f}".format(t_value)))
    
    print('p-value for two tailed test is %f'%p_value)
    
def perform_t_test_3_group(d_0, d_1, d_2):
    print("###############################################")
    print("Between 0 and 1")
    perform_t_test(d_0, d_1)
    print("Between 0 and 2")
    perform_t_test(d_0, d_2)
    print("Between 1 and 2")
    perform_t_test(d_1, d_2)
    print("###############################################")

def perform_welch_annova(d_0, d_1, d_2):
    pg.welch_anova(dv='score', between='group', data=df)

def perform_kruskal_test_at_once(d_0, d_1, d_2):
    t,p = stats.kruskal(d_0, d_1, d_2)
    print('p-value for two tailed test is %f'%p)

          
def perform_kruskal_test_repeated(d_0, d_1, d_2):
    print("###############################################")
    print("Between 0 and 1")
    t,p = stats.kruskal(d_0, d_1)
    print('p-value for two tailed test is %f'%p)
    print("Between 0 and 2")
    t,p = stats.kruskal(d_0, d_2)
    print('p-value for two tailed test is %f'%p)
    print("Between 1 and 2")
    t,p = stats.kruskal(d_1, d_2)
    print('p-value for two tailed test is %f'%p)
    print("###############################################")
          
def post_hoc_analysis_tukey(df):
    tukey = pairwise_tukeyhsd(endog=df['score'],
                          groups=df['group'],
                          alpha=0.05)
    print(tukey)

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

def PCA_based_stats():
    dataset = pd.read_csv('dataset_pca.csv')

    X = dataset.iloc[:,5:]
    Y = dataset.iloc[:,3]
    
    d_0 = X[Y==0]
    d_1 = X[Y==1]
    d_2 = X[Y==2]
    
    no_PCA_components = 5
    i = 0
    while i < no_PCA_components:
        print("###############################################")
        print("performing welch test for " + str(i+1) + "  PCA")
        print("Between 0 and 1")
        perform_t_test(d_0.iloc[:,i], d_1.iloc[:,i])
        print("Between 0 and 2")
        perform_t_test(d_0.iloc[:,i], d_2.iloc[:,i])
        print("Between 1 and 2")
        perform_t_test(d_1.iloc[:,i], d_2.iloc[:,i])
        print("###############################################")
        i += 1
    
    #X = dataset.iloc[:,3:5]
#    X["TrueClass"] = dataset.iloc[:,3]
    ax = sns.boxplot(x="pca_feature_0", y="TrueClass", data=dataset)

######################### main code starts here ###################################

#dataset = pd.read_csv ('master_dataset_GeMAPS.csv')

