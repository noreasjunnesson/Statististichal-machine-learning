import csv
from QDA import qda
from Random_forst import random_forest
from Logistic_regression import logistic_regression
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filename):
    np.random.seed(1)
    file_path = 'filename.csv'
    siren_data_notshuffle= pd.read_csv(file_path, na_values='?', dtype={'index': str}).dropna().reset_index()
    #shuffleing the data
    siren_data = siren_data_notshuffle.sample(frac=1).reset_index(drop=True)

    #creating input variable distance
    y_coor=siren_data.ycoor
    x_coor=siren_data.xcoor
    near_x=siren_data.near_x
    near_y=siren_data.near_y
    coord=np.array([x_coor,y_coor])
    near= np.array([near_x, near_y])
    # Calculate Euclidean distances for each pair
    distances = np.sqrt((x_coor - near_x)**2 + (y_coor - near_y)**2)
    #heard är x^2 beroende av distance
    distances_2=distances**2
    #tar bort inputs som används till distances
    columns_to_drop = ['ycoor', 'xcoor', 'near_x','near_y','near_fid']
    siren_data = siren_data.drop(columns=columns_to_drop)
    #lägger på distances^2
    siren_data['distance'] = distances_2


    # Tar absolutvärdet av vinkeln
    column_to_abs = 'near_angle'
    # Lägger in absolut belopp
    siren_data[column_to_abs] = siren_data[column_to_abs].abs()

    # Kolumner att standarisera (ej binära kolumner)
    columns_to_standardize = ['distance','near_angle','age']
    # StandardScaler 
    scaler = StandardScaler()
    # anpassa standardaserade kolumner
    siren_data[columns_to_standardize] = scaler.fit_transform(siren_data[columns_to_standardize])

    #np.random.seed(1)
    #splittar upp till 75% training and validation och 25% test
    trainIndex=np.random.choice(siren_data.shape[0],size=int(len(siren_data)*0.75),replace=False)
    
    #test and train
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

    # Train models
    trained_model_QDA = model_QDA.train(X_train,Y_train)
    trained_model_Random_forest = model_Random_forest.train(X_train,Y_train)
    trained_model_Logistic_regression=model_Logistic_regression(X_train,Y_train)


    #Predictions models
    prediction_QDA=trained_model_QDA.predict(X_test)
    prediction_Random_forest=trained_model_Random_forest.predict(X_test)
    prediction_Logistic_regression=trained_model_Logistic_regression.predict(X_test)
    
    #Accuray of models
    accuray_QDA=np.mean(prediction_QDA == Y_test)
    accuray_Random_forest=np.mean(prediction_Random_forest == Y_test)
    accuray_Logistic_regression=np.mean(prediction_Logistic_regression == Y_test)


    # Compare results or perform further analysis
    # For example:
    print("Accuracy QDA:", accuray_QDA)
    print("Accuracy Random forest:", accuray_Random_forest)
    print("Accuracy Logistic regression:", accuray_Logistic_regression)

if __name__ == "__main__":
    main()
