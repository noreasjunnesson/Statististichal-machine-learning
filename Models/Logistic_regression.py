import csv
from QDA import qda
from Random_forest import random_forest
from Logistic_regression import logistic_regression
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

class logistic_regression:
  def train(self, X_train, Y_train):
    n_fold = 10
    cv = KFold(n_splits=n_fold, random_state=1, shuffle=True)
    log_reg = skl_lm.LogisticRegression(max_iter=10000)
    log_reg.fit(X_train, Y_train)
    param_grid = {
    'C': np.logspace(-3, 3, 7),  # regularization parameter
    'solver': ['lbfgs'],  # optimization algorithm
    'penalty': ["l1","l2"] #l1 lasso, l2 ridge
    grid_search = GridSearchCV(log_reg, param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    trained_log_reg = grid_search.best_estimator_
    return trained_log_reg

