import csv
import sklearn.discriminant_analysis as skl_da
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score,recall_score,precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import sklearn.discriminant_analysis as skl_da
from sklearn.neighbors import KNeighborsClassifier


class k_nearest:
  def train(self, X_train, Y_train):
    n_fold = 10
    cv = KFold(n_splits=n_fold, random_state=1, shuffle=True)
    knn = KNeighborsClassifier(n_neighbors=1)
    k_range=list(range(1,31))
    param_grid = {
    'n_neighbors': k_range,
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
     }
    grid_search = GridSearchCV(knn, param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    trained_knn = grid_search.best_estimator_
    return trained_knn



