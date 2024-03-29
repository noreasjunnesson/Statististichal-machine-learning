import csv
import sklearn.discriminant_analysis as skl_da
from QDA import qda
from Random_forest import random_forest
from Logistic_regression import logistic_regression
from Naive import naive
from K_nearest import k_nearest
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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
from sklearn.neighbors import KNeighborsClassifier


def load_data(filename):
    np.random.seed(1)
    file_path = filename
    siren_data_notshuffle= pd.read_csv(file_path, na_values='?', dtype={'index': str}).dropna().reset_index()
    #shuffleing the data
    siren_data = siren_data_notshuffle.sample(frac=1).reset_index(drop=True)

    # Creating input variable distance
    y_coor=siren_data.ycoor
    x_coor=siren_data.xcoor
    near_x=siren_data.near_x
    near_y=siren_data.near_y
    coord=np.array([x_coor,y_coor])
    near= np.array([near_x, near_y])
    
    # Calculate Euclidean distances for each pair
    distances = np.sqrt((x_coor - near_x)**2 + (y_coor - near_y)**2)

    #Drop the columns associated with coordianats
    columns_to_drop = ['near_x', 'near_y', 'xcoor','ycoor', 'near_fid']
    siren_data = siren_data.drop(columns=columns_to_drop)
    
    #Adding 'distances' to the features
    siren_data['distances'] = distances

    #Squaring the age
    siren_data['age'] = siren_data.age**2

    #Taking absolut value of the near angle
    siren_data['angle'] = abs(siren_data.near_angle)

    #Dropping columns again
    columns_to_drop2 = ['near_angle','building','no_windows','noise']
    siren_data = siren_data.drop(columns=columns_to_drop2)

    #Scaling
    features_to_scale = ['distances','angle','age']
    selected_features = siren_data[features_to_scale]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(selected_features)
    siren_data[features_to_scale] = scaled_features

    # Splitting the data into 75% for training and validation and for 25% testing
    trainIndex=np.random.choice(siren_data.shape[0],size=int(len(siren_data)*0.75),replace=False)
    
    # Test and Train
    train=siren_data.iloc[trainIndex]
    test=siren_data.iloc[~siren_data.index.isin(trainIndex)]

    X_train=train.copy().drop(columns='heard')
    Y_train=train['heard']
    X_test=test.copy().drop(columns='heard')
    Y_test=test['heard']
    
    return X_train,Y_train,X_test,Y_test

def main():
    # Load data
    X_train,Y_train,X_test,Y_test = load_data("siren_data_train.csv")

    # Instantiate models
    model_QDA = qda()
    model_Random_forest = random_forest()
    model_Logistic_regression=logistic_regression()
    model_naive=naive()
    model_knn=k_nearest()

    # Train models
    trained_model_QDA = model_QDA.train(X_train,Y_train)
    trained_model_Random_forest = model_Random_forest.train(X_train,Y_train)
    trained_model_Logistic_regression=model_Logistic_regression.train(X_train,Y_train)
    trained_model_knn = model_knn.train(X_train,Y_train)

    #Predictions models
    prediction_QDA=trained_model_QDA.predict(X_test)
    prediction_Random_forest=trained_model_Random_forest.predict(X_test)
    prediction_Logistic_regression=trained_model_Logistic_regression.predict(X_test)
    prediction_naive=model_naive.predict(X_test)
    prediction_knn=trained_model_knn.predict(X_test)
  
    #Accuray of models
    accuray_QDA=np.mean(prediction_QDA == Y_test)
    accuray_Random_forest=np.mean(prediction_Random_forest == Y_test)
    accuray_Logistic_regression=np.mean(prediction_Logistic_regression == Y_test)
    accuray_naive=np.mean(prediction_naive == Y_test)
    accuray_knn=np.mean(prediction_knn == Y_test)
    
    #F1 score of models
    f1_QDA = f1_score(Y_test,prediction_QDA)
    f1_Random_forest = f1_score(Y_test,prediction_Random_forest)
    f1_Logistic_regression = f1_score(Y_test,prediction_Logistic_regression)
    f1_naive = f1_score(Y_test,prediction_naive)
    f1_knn = f1_score(Y_test,prediction_knn)

    #Recall score of models
    recall_QDA = recall_score(Y_test,prediction_QDA)
    recall_Random_forest = recall_score(Y_test,prediction_Random_forest)
    recall_Logistic_regression = recall_score(Y_test,prediction_Logistic_regression)
    recall_naive = recall_score(Y_test,prediction_naive)
    recall_knn = recall_score(Y_test,prediction_knn)

    #Precision score of models
    precision_QDA = precision_score(Y_test,prediction_QDA)
    precision_Random_forest = precision_score(Y_test,prediction_Random_forest)
    precision_Logistic_regression = precision_score(Y_test,prediction_Logistic_regression)
    precision_naive = precision_score(Y_test,prediction_naive)
    precision_knn = precision_score(Y_test,prediction_knn)
 
    data = {
    'Model': ['QDA', 'Random Forest', 'Logistic Regression','KNN','Naive'],
    'Accuracy': [accuray_QDA, accuray_Random_forest, accuray_Logistic_regression,accuray_knn ,accuray_naive],
    'F1 Score': [f1_QDA, f1_Random_forest, f1_Logistic_regression,f1_knn, f1_naive],
    'Recall': [recall_QDA, recall_Random_forest, recall_Logistic_regression,recall_knn, recall_naive],
    'Precision': [precision_QDA, precision_Random_forest, precision_Logistic_regression,precision_knn, precision_naive]
    }

    df = pd.DataFrame(data)

    # Print the DataFrame
    print(df)
    # Print the DataFrame
    print(df)




if __name__ == "__main__":
    main()
