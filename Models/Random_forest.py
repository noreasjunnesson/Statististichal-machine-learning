
class random_forest:
    def train(self, X_train,Y_train):

        
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

        

        def grid_search(X_train,Y_train):
            param_grid = {'n_estimators': [100],
                        'max_depth': [ 5, 10,20],
                        'criterion': ['entropy'],
                        'min_samples_split': [3,4],
                        'min_samples_leaf': [3,4],
                        'max_features':[2,3,4],}
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1)
            grid_search.fit(X_train, Y_train)
            return grid_search.fit(X_train, Y_train)
            
    trained_Random_forest=grid_search(X_train,Y_train)
    
    return trained_Random_forest
