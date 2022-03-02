# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 11:06:43 2021

@author: bhupendra.singh
"""

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random

featuresFolder = ".//WebBasedTest/features_files"
targetClassFoler = featuresFolder = ".//WebBasedTest/classes"

totalSubjects = 21
totalUtterances = 60    #the number of utterances of words in a folder for every subject
train_test_per = .70     # the number of percentage given to training set
featureName = "mfcc"
features_length = 6373
#features_length = 128

train_target_set = list()
test_target_set = list()

utterancesFolder = ".//WebBasedTest/word_level_utterances"
featuresFolder = ".//WebBasedTest/features_files"

availableSubjectList = list(range(totalSubjects))
availableSubjectList = [x+1 for x in availableSubjectList]  # adding 1 to all the itesms since the subject folder starts from 1 and not 0
#availableSubjectList.remove(4)

#avialableSubjectList = [11]
def shuffle_list(input_list):
    # using Fisher–Yates shuffle Algorithm
    # to shuffle a list
    for i in range(len(input_list)-1, 0, -1):
     
        # Pick a random index from 0 to i
        j = random.randint(0, i + 1)
   
        # Swap arr[i] with the element at random index
        input_list[i], input_list[j] = input_list[j], input_list[i]
        
    return input_list

tmp_list = shuffle_list(availableSubjectList.copy())
index_of_split = int(len(tmp_list)*train_test_per)
train_subject_list = tmp_list[0:index_of_split]
test_subject_list = tmp_list[index_of_split:]
#print(tmp_list) 

train_feature_set = np.zeros((len(train_subject_list)*totalUtterances, features_length))
test_feature_set = np.zeros((len(test_subject_list)*totalUtterances, features_length))

### generate the training data ##############
count = 0
for currentSubjectNo in tqdm(train_subject_list):     # iterate for all the available subjects\
    targetFileName = targetClassFoler + "/Subject" + str(currentSubjectNo) + ".xlsx"
    featureFileName = featuresFolder + "/Subject" + str(currentSubjectNo) + "_" + featureName + ".csv"
    
    df1 = pd.DataFrame(pd.read_csv(featureFileName))
    features_per_subject = df1.iloc[:,1:]
    for i in range(features_per_subject.shape[0]):
        train_feature_set[count, :] = features_per_subject.iloc[i,:]
        count += 1
    df = pd.DataFrame(pd.read_excel(targetFileName))
    classes = df.iloc[ : totalUtterances, 1]
    #print("Current Subject = " + str(currentSubjectNo) + " and total classes = " + str(len(classes)))
    train_target_set.extend(classes)

df = pd.DataFrame(train_feature_set)
df2 = pd.DataFrame(train_target_set)
df2.to_csv("train_target.csv")
df.to_csv("train_features.csv")
###########################################

### generate the testing data ##############
count = 0
for currentSubjectNo in tqdm(test_subject_list):     # iterate for all the available subjects\
    targetFileName = targetClassFoler + "/Subject" + str(currentSubjectNo) + ".xlsx"
    featureFileName = featuresFolder + "/Subject" + str(currentSubjectNo) + "_" + featureName + ".csv"
    
    df1 = pd.DataFrame(pd.read_csv(featureFileName))
    features_per_subject = df1.iloc[:,1:]
    for i in range(features_per_subject.shape[0]):
        test_feature_set[count, :] = features_per_subject.iloc[i,:]
        count += 1
    df = pd.DataFrame(pd.read_excel(targetFileName))
    classes = df.iloc[ : totalUtterances, 1]
    #print("Current Subject = " + str(currentSubjectNo) + " and total classes = " + str(len(classes)))
    test_target_set.extend(classes)

df = pd.DataFrame(test_feature_set)
df2 = pd.DataFrame(test_target_set)
df2.to_csv("test_target.csv")
df.to_csv("test_features.csv")
###########################################

#plot_pie_class_distribution(target_set)

"""
for currentSubjectNo in tqdm(availableSubjectList):     # iterate for all the available subjects\
    featuresFileName = featuresFolder + "/Subject" + str(currentSubjectNo) + "_" + featureName + ".csv"
print(featuresFileName)
#df = pd.read_csv(featuresFileName)
"""
## Temperary code for age distribution plot
y = np.array([16, 12, 8, 3])
mylabels = ["18-20", "21-30", "31-40", "41-50"]
plt.pie(y, labels = mylabels, autopct='%.2f')
plt.legend(title = "Age Distributions")
plt.savefig("age_distribution.png", dpi=300)
plt.show()
    

def plot_pie_class_distribution(target_set):    
    class_0 = [x for x in target_set if x == 0]
    class_0 = len(class_0)
    class_1 = [x for x in target_set if x == 1]
    class_1 = len(class_1)
    class_2 = [x for x in target_set if x == 2]
    class_2 = len(class_2)

    y = np.array([class_0, class_1, class_2])
    mylabels = ["No Stress", "Low Stress", "High Stress"]
    plt.pie(y, labels = mylabels, autopct='%.2f')
    plt.legend(title = "Class Distributions")
    plt.savefig("classes_distribution.png", dpi=300)
    plt.show()

    print(class_0+class_1+class_2)
