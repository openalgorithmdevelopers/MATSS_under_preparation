# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:34:48 2021

@author: bhupendra.singh
"""

# compare performance on the provided dataset using various ML models ############
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
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
#Y[Y==1] = 0
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
models.append(('GMM', mixture.GaussianMixture(n_components=1, covariance_type='full')))
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
#plt.boxplot(results, labels=names)
#plt.title('Algorithm Comparison')
#plt.show()


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import svm

x_rus, y_rus = rus.fit_resample(X, Y)
x_train, x_test, y_train, y_test = train_test_split(x_rus, y_rus, test_size=0.3, random_state=42)

clf = svm.SVC(kernel='linear', C=1, random_state=0)


clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
#y_pred = cross_val_predict(clf, x_test, y_test, cv=kfold)
#score = cross_val_score(clf, x_rus, y_rus, cv=kfold, scoring='accuracy')
cm = confusion_matrix(y_test, y_pred)
#print(cm)
from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(y_test, y_pred)
from sklearn.metrics import accuracy_score
sc = accuracy_score(y_test, y_pred)
print(sc)
#print(score.mean())

#x_train, x_test, y_train, y_test = train_test_split(x_rus, y_rus, test_size=0.2, random_state=42)
#skf = StratifiedKFold(n_splits=10)
#model = SVC(gamma='auto')
#model.fit(x_train, y_train)
#score = model.score(x_test,y_test)

def ROC():
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    
    X_train = x_train
    X_test = x_test
    # fit model
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    pred_prob = clf.predict_proba(X_test)
    
    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh ={}
    
    n_class = 3
    
    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:,i], pos_label=i)
        
    # plotting    
    plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class 0 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Class 1 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Class 2 vs Rest')
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('Multiclass ROC',dpi=300);  