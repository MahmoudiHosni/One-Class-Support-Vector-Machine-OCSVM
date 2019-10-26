#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 07:01:55 2018

@author: hosni
"""
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
from sklearn.model_selection import train_test_split

data1=[]
with open("/home/hosni/Desktop/Classe1.csv") as file:
    fichier1=csv.reader(file,delimiter=",")
    #next(fichier1)
    for i in fichier1:
        data1.append(i)
 
#print(data1)
#data1=1
data2=[]
with open("/home/hosni/Desktop/Classe2.csv") as file:
    fichier2=csv.reader(file,delimiter=",")
    #next(fichier2)
    for i in fichier2:
        data2.append(i)
#data1_train,data1_test=train_test_split(data1,test_size=0.33)

train=data1_train

test=np.concatenate((data1_test,data2),axis=0)
#print(test)

#labels 
labeltest1=np.ones((165,1))
labeltest2=np.ones((500,1))*-1
print(len(labeltest2))
print(len(labeltest1))
labeltest=np.concatenate((labeltest1,labeltest2),axis=0)
print(labeltest)
clf = svm.OneClassSVM(nu=0.1,kernel='rbf',gamma =0.09)
#clf = svm.OneClassSVM(nu=0.1,kernel='linear', gamma=0.09)
#clf = svm.OneClassSVM(nu=0.1,kernel='poly', coef0=5 gamma=0.09)
#clf = svm.OneClassSVM(nu=0.1,kernel='sigmoid', coef0=1, gamma=0.001)

clf.fit(train)
predicted=clf.predict(test)
print("Predicted =",len(predicted))
accuracy=accuracy_score(labeltest,predicted)
print("accuracy ",accuracy)
fpr, tpr, thresholds=roc_curve(labeltest, predicted, pos_label=-1)
print("fpr= ",fpr)
print("tpr= ",tpr)
print("thresholds= ",thresholds)
plt.plot(tpr,fpr)
plt.title("roc curve")
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
print("auc= ",auc(tpr,fpr))
print("Matrice de confusion",confusion_matrix(labeltest, predicted, labels=None, sample_weight=None))
