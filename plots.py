# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 18:27:31 2022

@author: bhupendra.singh
"""
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
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import svm
import os

from data_analysis import *

sns.set(rc={"figure.dpi":100, 'savefig.dpi':400})


dataset_PCA = pd.read_csv('dataset_pca.csv')
dataset_PCA['TrueClass'] = dataset_PCA['TrueClass'].astype(int)
dataset_Master = pd.read_csv('master_dataset.csv')

def plotScatter():
    ######## Scatter plot ####################
    fig, axs = plt.subplots(nrows=2)
    
    ax1 = sns.scatterplot(data=dataset_PCA, x="SampleNo", y="pca_component_1", hue="TrueClass", palette="deep", ax=axs[0])
    ax2 = sns.scatterplot(data=dataset_PCA, x="SampleNo", y="pca_component_2", hue="TrueClass", palette="deep", ax=axs[1])
    ax1.set(xlabel='Sample No.',
           ylabel='1st Principal Comp.')
    ax2.set(xlabel='Sample No.',
           ylabel='2nd Principal Comp.')
    #fig = pt.get_figure()
    fig.savefig('pca_scatter.png')

def plotBox():
    ####### Box plot ####################
    pt, axs = plt.subplots(nrows=2)
    ax3 = sns.boxplot(x="TrueClass", y='pca_component_1', data=dataset_PCA,  ax=axs[0])
    ax4 = sns.boxplot(x="TrueClass", y='pca_component_2', data=dataset_PCA,  ax=axs[1])
    ax3.set(xlabel='True Class',
           ylabel='1st Principal Comp.')
    ax4.set(xlabel='True Class',
           ylabel='2nd Principal Comp.')
    pt.savefig('pca_box.png')
    
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
    plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='No Stress vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Low Stress vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='High Stress vs Rest')
#    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('ROC_plot',dpi=300); 
    
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.grid(False)
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('Confustion_Matrix',dpi=300); 