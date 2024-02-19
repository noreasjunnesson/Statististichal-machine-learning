import sklearn.discriminant_analysis as skl_da
from sklearn.model_selection import GridSearchCV

class qda:
  def train(self, X_train, Y_train):
    qda = skl_da.QuadraticDiscriminantAnalysis() #implement the qda model
    qda.fit(X_train, Y_train) #fit the training data to the qda model
    param_grid = {'reg_param': np.linspace(0.0, 0.2,num=25)}  #the parameter values often range from 0 to 1, with 0 meaning no regularization (QDA becomes LDA) and values closer to 1 indicating stronger regularization
    #tuning the parameter grid showed that the best accuracy was achieved with lower regularization parameters
    grid_search = GridSearchCV(qda, param_grid, cv=5, scoring='accuracy') #tune metod using grid search cross validation
    grid_search.fit(X_train, Y_train) #fit the grid search to the training data
    best_params = grid_search.best_params_  #get the optimal values for the regularization parameter
    trained_qda = grid_search.best_estimator_ #returns the best estimator (model) that was found during the hyperparameter tuning process
    return trained_qda
