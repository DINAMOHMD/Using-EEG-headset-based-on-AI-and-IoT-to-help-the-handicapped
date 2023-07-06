# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 08:28:35 2023

@author: pc
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
convert_Data=pd.read_excel('C:/Users/pc/.spyder-py3/BCI Project/BCI_DATA3_4.xlsx')
X=convert_Data.iloc[:,:-1].values
y=convert_Data.iloc[:,-1].values

#----------------------------------------------------
#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)
#----------------------------------------------------
KNNClassifierModel = KNeighborsClassifier(n_neighbors= 5,weights ='uniform', algorithm='auto') 
KNNClassifierModel.fit(X_train, y_train)
#Calculating Score
print('KNNClassifierModel Train Score is : ' , KNNClassifierModel.score(X_train, y_train))
print('KNNClassifierModel Test Score is : ' , KNNClassifierModel.score(X_test, y_test))
#Calculating Prediction
y_pred = KNNClassifierModel.predict(X_test)
y_pred_prob = KNNClassifierModel.predict_proba(X_test)
# Compute predicted labels for test data
y_pred = KNNClassifierModel.predict(X_test)
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', cm)
# Plot confusion matrix
sns.heatmap(cm)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix')
plt.show()
# Convert labels to one-hot encoded vectors
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
# Compute probabilities for each class
probas = KNNClassifierModel.predict_proba(X_test)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probas[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), probas.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Plot ROC curves
plt.figure()

lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
         lw=lw, label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(5):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()
AccScore=accuracy_score(y_test,y_pred)
print('Accuracy Score is:',AccScore)
ZeroOneLossValue=zero_one_loss(y_test, y_pred)
print('Misclassification Rate:',ZeroOneLossValue)