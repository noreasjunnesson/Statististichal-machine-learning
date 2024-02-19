#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:10:26 2024

@author: ebbalorin
"""

import numpy as np
import numpy.linalg as nplg
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplg
from numpy.linalg import norm

import pandas as pd
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms


from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score

#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('png')
from IPython.core.pylabtools import figsize
figsize(10, 6) # Width and hight
#plt.style.use('seaborn-white')


data_notshuffled=pd.read_csv('siren_data_train.csv', na_values='?', dtype={'ID':str}).dropna().reset_index()
data=data_notshuffled.sample(frac=1).reset_index(drop=True)

#sumsquares = np.sum(np.power((data.xcoor-data.ycoor),2))

# take the square root of the sum of squares to obtain the L2 norm
#l2_norm = np.sqrt(sumsquares)
#print(l2_norm)

#plt.rcParams["figure.autolayout"] = True

#Take the Euklidian distance:
distance=np.sqrt((data.xcoor-data.near_x)**2 + (data.ycoor-data.near_y)**2)

xcoor=data.xcoor
ycoor=data.ycoor
near_x=data.near_x
near_y=data.near_y

coord=np.array([xcoor,ycoor])
near=np.array([near_x,near_y])

columnsdrop=['near_x','near_y','xcoor','ycoor']
data=data.drop(columns=columnsdrop)

data['distance']=distance

#Split data into training and testing dataset:

trainI = np.random.choice(data.shape[0],size= int(len(data)*0.75),replace=False)
trainIndex = data.index.isin(trainI)
train = data.iloc[trainIndex]
test = data.iloc[~trainIndex]

X_train = train.copy().drop(columns='heard')
Y_train = train['heard']
X_test = test.copy().drop(columns='heard')
Y_test = test['heard']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#k-NN with k=1
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, Y_train)

knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, Y_train)

prediction1=knn1.predict(X_test)
prediction5=knn5.predict(X_test)
#print(pd.crosstab(prediction1,Y_test),'\n')
#Accuracy
#print(f"Accuracy: {np.mean(prediction == Y_test):.3f}")
#print("Accuracy ", accuracy_score(Y_test,prediction1))


# Confusion Matrix for the testing set
#cm = confusion_matrix(Y_test, prediction1)

#cm_display = ConfusionMatrixDisplay(cm).plot()

# ROC Curve
fpr, tpr, thresholds = roc_curve(Y_test, prediction1)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
#plt.figure(figsize=(8, 8))
#plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC) Curve')
#plt.legend(loc='lower right')
#plt.show()

#Trying different values for k

misclassification=[]
accuracy=[]
for k in range(50):
    model=skl_nb.KNeighborsClassifier(n_neighbors=k+1)
    model.fit(X_train, Y_train)
    prediction=model.predict(X_test)
    misclassification.append(np.mean(prediction !=Y_test))
    ac=accuracy_score(Y_test,prediction)
    accuracy.append(ac)

K=np.linspace(1,50,50)
#plt.plot(K,misclassification,'.')
#plt.ylabel('Misclassification rate')
#plt.xlabel('Numer of neighbors')
#plt.show()   

# Confusion Matrix

#cm = confusion_matrix(Y_test, prediction)

#cm_display = ConfusionMatrixDisplay(cm).plot()

plt.plot(K,accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Number of neighbors')
plt.grid(True)
plt.show()
