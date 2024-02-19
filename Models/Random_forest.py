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
class random_forest:
    def train(self, X_train,Y_train):

        model = RandomForestClassifier()
        n_fold = 10
        cv = KFold(n_splits=n_fold, random_state=1, shuffle=True)
        param_grid = {'n_estimators': [100],
                        'max_depth': [ 5, 10,20],
                        'criterion': ['entropy'],
                        'min_samples_split': [3,4],
                        'min_samples_leaf': [3,4],
                        'max_features':[2,3,4],}
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        trained_Random_forest=grid_search.best_estimator_
        return trained_Random_forest
    
