class logistic_regression:
  def train(self, X_train, Y_train):
    log_reg = skl_lm.LogisticRegression(max_iter=10000)
    log_reg.fit(X_train, Y_train)
    param_grid = {
    'C': np.logspace(-3, 3, 7),  # regularization parameter
    'solver': ['lbfgs'],  # optimization algorithm
    'penalty': ["l1","l2"] #l1 lasso, l2 ridge
    grid_search = GridSearchCV(log_reg, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    trained_log_reg = grid_search.best_estimator_
    return trained_logreg

