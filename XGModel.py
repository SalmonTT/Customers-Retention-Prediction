from dataPreprocessing import *
from sklearn.model_selection import train_test_split, GridSearchCV
from utils import print_score, ROC, printFullRow
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE

def xgChan():
    # https://goodboychan.github.io/chans_jupyter/python/datacamp/machine_learning/2020/07/07/02-Fine-tuning-your-XGBoost-model.html
    train = getTrainingData('train.csv', visualize=False)
    X = train.drop(['Exited'], axis=1)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = train.Exited
    matrix = xgb.DMatrix(data=X, label=y)
    params = {
        "objective": "reg:squarederror",
        "max_depth": 3
    }
    # ----- manually config number of rounds ----
    # num_rounds = [5,10,15]
    # # Empty list to store final round rmse per XGBoost model
    # final_rmse_per_round = []
    # # Interate over num_rounds and build one model per num_boost_round parameter
    # for curr_num_rounds in num_rounds:
    #     # Perform cross-validation: cv_results
    #     cv_results = xgb.cv(dtrain=matrix, params=params, nfold=3,
    #                         num_boost_round=curr_num_rounds, metrics='rmse',
    #                         as_pandas=True, seed=123)
    #
    #     # Append final round RMSE
    #     final_rmse_per_round.append(cv_results['test-rmse-mean'].tail().values[-1])
    # # revision on RMSE (Root Mean Square Error): differences between values predicted by the model and the actual values
    # # lowest RMSE = 0 (perfect model)
    # # Print the result DataFrame
    # num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
    # print(pd.DataFrame(num_rounds_rmses, columns=['num_boosting_rounds', 'rmse']))

    # ---- auto stopping ---
    # Perform cross-validation with early-stopping: cv_results
    # note that the largest possible number of boosting rounds == 50 here
    cv_results = xgb.cv(dtrain=matrix, nfold=3, params=params, metrics="rmse",
                        early_stopping_rounds=10, num_boost_round=50, as_pandas=True, seed=123)

    # Print cv_results
    print(cv_results)


    return


def xgModel():
    train = getTrainingData('train.csv', visualize=False)
    X = train.drop(['Exited'], axis=1)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = train.Exited
    # method 1: technically only for continuous data
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    # ----- random_state is a seed for random sampling -----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # ----- standardizing the input features -----
    X_train, X_test = standard(X_train, X_test)
    model = xgb.XGBClassifier()

    params = {
        'n_estimators': [10,25,70],
        'max_depth': [6,7,8],
        'gamma': [1, 0.05, 0.1]

    }

    grid_search_cv = GridSearchCV(model, params, verbose=1, n_jobs=-1, cv=3, scoring='f1')
    grid_search_cv.fit(X_train, y_train)
    best_grid = grid_search_cv.best_estimator_
    print(best_grid)
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=False)

    return
xgModel()
# xgChan()