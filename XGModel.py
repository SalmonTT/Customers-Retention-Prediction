from dataPreprocessing import *
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from utils import print_score, ROC, printFullRow
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE
from imblearn.combine import SMOTEENN


def xgModelNew():
    X_train, y_train, X_val, y_val, test_train, test_val = getAllCleanedData(standardize=0, binning=1)
    printFullDf(X_train.head())

    model = xgb.XGBClassifier()
    params = {
        'n_estimators': [15, 20],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
        'max_depth': [4, 5, 6],
        'gamma': [0, 0.1, 0.3 , 0.5, 0.7, 1],
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
        "min_child_weight": [1, 3, 5, 7],
        'validate_parameters': [0]
    }

    grid_search_cv = RandomizedSearchCV(model, params, verbose=1, n_jobs=5, cv=5, scoring='f1')
    grid_search_cv.fit(X_train, y_train)
    best_grid = grid_search_cv.best_estimator_
    print(best_grid)
    print_score(grid_search_cv, X_train, y_train, X_val, y_val, train=True)
    print_score(grid_search_cv, X_train, y_train, X_val, y_val, train=False)
    print_score(grid_search_cv, X_train, y_train, test_train, test_val, train=False)

def tuning():
    X_train, y_train, test_train, test_val = getAllCleanedDataBig(standardize=1)
    printFullDf(X_train.head())

    testing_params = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    RFtrainAcc = []
    RFtestAcc = []
    for param in testing_params:
        xgb_classifier = xgb.XGBClassifier(max_depth=4, min_child_weight=5, n_estimators=20,
                                           learning_rate=0.22, colsample_bytree=0.45)
        rf = xgb_classifier
        rf.fit(X_train, y_train)
        y_predTrain = rf.predict(X_train)
        y_predTest = rf.predict(test_train)
        RFtrainAcc.append(f1_score(y_train, y_predTrain))
        RFtestAcc.append(f1_score(test_val, y_predTest))
    print_score(xgb_classifier, X_train, y_train, test_train, test_val, train=False)
    plt.plot(testing_params, RFtrainAcc, 'ro-', testing_params, RFtestAcc, 'bv--')
    plt.legend(['Training f1', 'Test f1'])
    plt.xlabel('n_estimators')
    # plt.xscale('log')
    plt.ylabel('f1')
    plt.show()

# tuning()
xgModelNew()
