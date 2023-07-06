# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 14:51:56 2023

@author: pc
"""

import pandas as pd
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
convert_Data=pd.read_excel('C:/Users/pc/.spyder-py3/BCI Project/BCI_DATA3_4.xlsx')
X=convert_Data.iloc[:,:-1].values
y=convert_Data.iloc[:,-1].values
#----------------------------------------------------
#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)
#----------------------------------------------------
clf = OneVsOneClassifier(SVC(kernel='linear', C=1))
# train the classifier on training data
clf.fit(X_train, y_train)
# test the classifier on testing data
y_pred = clf.predict(X_test)
# evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
