# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 08:22:20 2021

@author: bhupendra.singh
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment
from python_speech_features import mfcc, logfbank 
import opensmile

import warnings
warnings.filterwarnings('ignore')


totalSubjects = 21
totalUtterances = 60    #the number of utterances of words in a folder for every subject
featureName = "mfcc"
FEATURE_SIZE = 128
FEATURE_SIZE_GeMAPS = 62

#utterancesFolder = ".//WebBasedTest/word_level_utterances"
utterancesFolder = "./word_level_utterances_manual_refined"
#featuresFolder = "/features_files"

avialableSubjectList = list(range(totalSubjects))
avialableSubjectList = [x+1 for x in avialableSubjectList]  # adding 1 to all the itesms since the subject folder starts from 1 and not 0
#avialableSubjectList.remove(4)

totalUtterancesList = list(range(totalUtterances))
totalUtterancesList = [x+1 for x in totalUtterancesList]  # adding 1 to all the itesms since the subject folder starts from 1 and not 0

#features_set = np.zeros((len(totalUtterancesList), FEATURE_SIZE))
features_set = np.zeros((len(totalUtterancesList), FEATURE_SIZE_GeMAPS))
fs = 48000  #i found it to be this value, change it as per your information

true_classes = pd.read_csv('Classes_all.csv')

#raw_results = np.zeros((1260, 4+FEATURE_SIZE))
raw_results = np.zeros((1260, 4+FEATURE_SIZE_GeMAPS))
sample = 1
for currentSubjectNo in tqdm(avialableSubjectList):     # iterate for all the available subjects
    #currentSubjectNo += 1
    print("Current Subject = " + str(currentSubjectNo))
    for currentUtteranceNo in totalUtterancesList: #iterate for for all the utterances
        #print("Current Subject = " + str(currentUtteranceNo))
        #utteranceFileName = utterancesFolder + "/utterances_subject_" + str(currentSubjectNo) + "/utterance" + str(currentUtteranceNo) + ".wav"
        utteranceFileName = utterancesFolder + "/utterances_subject_" + str(currentSubjectNo) + "/" + str(currentUtteranceNo) + ".wav"
        print(utteranceFileName)
        ########### For OpenSmile features
        
        smile = opensmile.Smile(
            #feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_set=opensmile.FeatureSet.GeMAPS,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        GeMaps_features = smile.process_file(utteranceFileName)
        #print(len(y))
        #mfcc_features.resize(128) #standardizing the size
        GeMaps_features = GeMaps_features.to_numpy()
        ############################
        
        sound_file = AudioSegment.from_wav(utteranceFileName)
        #mfcc_features = mfcc(np.array(sound_file.get_array_of_samples()), fs)
        
        #print("Printing the length of hte MFCC features" + str(len(mfcc_features)))
              
        #mfcc_features.resize(FEATURE_SIZE) #standardizing the size
        #features_set[currentUtteranceNo - 1, :] = mfcc_features
        features_set[currentUtteranceNo - 1, :] = GeMaps_features
        raw_results[sample-1, 0] = sample
        raw_results[sample-1, 1] = currentSubjectNo
        raw_results[sample-1, 2] = currentUtteranceNo
        raw_results[sample-1, 3] = true_classes.iloc[currentUtteranceNo-1, currentSubjectNo]
        #raw_results[sample-1, 4:] = mfcc_features
        raw_results[sample-1, 4:] = GeMaps_features
        sample += 1
column_names = list()
column_names.append('SampleNo')
column_names.append('Subject')
column_names.append('UtteranceNo')
column_names.append('TrueClass') 

i = 0
while i < FEATURE_SIZE_GeMAPS:
    column_names.append('feature_value_' + str(i)) 
    i += 1

df = pd.DataFrame(raw_results, columns = column_names)


#df.to_csv('./master_dataset.csv', index=False)
df.to_csv('./master_dataset_GeMAPS.csv', index=False)
