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
        def random_search(X_train, Y_train):
            param_dist = {'n_estimators': randint(50, 150),  # Random integer between 50 and 150
                            'max_depth': [None, 5, 10, 20],
                            'criterion': ['entropy', 'gini'],
                            'min_samples_split': randint(2, 6),  # Random integer between 2 and 6
                            'min_samples_leaf': randint(2, 5),   # Random integer between 2 and 4
                            'max_features': randint(2, 6),       # Random integer between 2 and 5
                             }
            n_iter_search = 20  # Number of parameter settings that are sampled
            random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=n_iter_search, cv=cv, n_jobs=-1)
            random_search.fit(X_train, Y_train)
            best_model = random_search.best_estimator_
            return print('With parameters:%2f:',best_model)

        

        def gridsearch(X_train,Y_train):
            param_grid = {'n_estimators': [100],
                        'max_depth': [ 5, 10,20],
                        'criterion': ['entropy'],
                        'min_samples_split': [3,4],
                        'min_samples_leaf': [3,4],
                        'max_features':[2,3,4],}
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1)
            grid_search.fit(X_train, Y_train)
            return grid_search.best_estimator_

        trained_Random_forest = gridsearch(X_train, Y_train)
        return trained_Random_forest
    
