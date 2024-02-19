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


#k-NN with k=1
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, Y_train)

knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, Y_train)

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




