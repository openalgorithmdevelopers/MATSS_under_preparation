# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:34:48 2021

@author: bhupendra.singh
"""

# compare performance on the provided dataset using various ML models ############
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import mixture
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

dataset_all = pd.read_csv ('master_dataset.csv')
Y = dataset_all.iloc[:,3]
X = dataset_all.iloc[:,4:]

rus = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
#x_rus, y_rus = rus.fit_resample(x_train, y_train)
x_rus, y_rus = rus.fit_resample(X, Y)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
models.append(('MLP', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)))
#models.append(('GMM', mixture.GaussianMixture(n_components=1, covariance_type='full')))
#models.append(('HMM', hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
cm =[]

for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, x_rus, y_rus, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

"""
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import svm

x_rus, y_rus = rus.fit_resample(x_train, y_train)
#clf = svm.SVC(kernel='linear', C=1, random_state=0)
clf = svm.SVC()
clf.fit(x_rus, y_rus)
#clf.predict(x_rus, y_test)
y_pred = cross_val_predict(clf, x_test, y_test, cv=10)
cm = confusion_matrix(y_test, y_pred)
print(cm)
"""
"""
skf = StratifiedKFold(n_splits=10)
model = SVC(gamma='auto')
model.fit(x_train, y_train)
score = model.score(x_test,y_test)
"""