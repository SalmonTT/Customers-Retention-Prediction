from dataPreprocessing import *
from sklearn.model_selection import train_test_split, GridSearchCV
from utils import print_score, ROC, printFullRow
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE

def xgModel():
    train = getTrainingData('train.csv', visualize=False, discrete=True, encoding=True)
    X = train.drop(['Exited'], axis=1)
    y = train.Exited
    # method 1: technically only for continuous data
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    # ----- random_state is a seed for random sampling -----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # ----- standardizing the input features -----
    scale = StandardScaler().fit(X_train)
    X_train = scale.transform(X_train)
    X_test = scale.transform(X_test)
    model = xgb.XGBClassifier()

    params = {
        'n_estimators': [60,70,80],
        'max_depth': [4,5,6],
        'gamma': [1],
        'learning_rate': [0.3,0.4,0.5]
    }

    grid_search_cv = GridSearchCV(model, params, verbose=1, n_jobs=-1, cv=5, scoring='f1')
    grid_search_cv.fit(X_train, y_train)
    best_grid = grid_search_cv.best_estimator_
    print(best_grid)
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    # ----- evaluate test sample and return a .csv ----
    test_data = getTestData('assignment-test.csv', False, True, True)
    # printFullDf(test_data)
    test_data = scale.transform(test_data)
    pred_prob = grid_search_cv.predict(test_data)
    print(pred_prob)
    exportCSV('assignment-test.csv', pred_prob)
    # evaluate_2.exe submission_2.csv
    return
xgModel()
