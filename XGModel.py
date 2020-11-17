from dataPreprocessing import *
from sklearn.model_selection import train_test_split, GridSearchCV
from utils import print_score, ROC, printFullRow
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE


def xgModel():
    train = getTrainingData('train.csv', visualize=False, discrete=True, encoding=True)
    X = train.drop(['Exited'], axis=1)
    y = train.Exited
    # ----- SMOTE ------
    # method 1: technically only for continuous data
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X, y)
    # method 2: SMOTE NC
    # smotenc = SMOTENC([4,6,7,8,9,10,11,12,13], random_state=101)
    # X_train, y_train = smotenc.fit_resample(X, y)

    # ----- random_state is a seed for random sampling -----
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = xgb.XGBClassifier()
    params = {
        'n_estimators': [80],
        'max_depth': [4],
        'gamma': [1],
        'learning_rate': [0.2]
    }

    grid_search_cv = GridSearchCV(model, params, verbose=1, n_jobs=-1, cv=5, scoring='f1')
    grid_search_cv.fit(X_train, y_train)
    best_grid = grid_search_cv.best_estimator_
    print(best_grid)
    # ----- get testing data ---- #
    df_test = getTestingData(True, True)
    X_test = df_test.drop(['Exited'], axis=1)
    y_test = df_test.Exited
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    # ----- export csv ------
    # test_data = getTestData('assignment-test.csv', False, True, True)
    # pred_prob = grid_search_cv.predict(test_data)
    # print(pred_prob)
    # exportCSV('assignment-test.csv', pred_prob)
    return

xgModel()
