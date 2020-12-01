from dataPreprocessing import *
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from utils import print_score, ROC, printFullRow
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from tensorflow.keras.callbacks import EarlyStopping
from deepLearning import kerasModel, get_f1
from sklearn.ensemble import VotingClassifier

def customEnsemble():
    X_train, y_train, X_val, y_val, test_train, test_val = getAllCleanedData(standardize=0)
    test_train_og = test_train
    test_val_og = test_val
    # --- categorical dataset
    X_trainCat = X_train[['NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Germany', 'Spain', 'Male']]
    X_valCat = X_val[['NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Germany', 'Spain', 'Male']]
    test_trainCat = test_train[['NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Germany', 'Spain', 'Male']]
    # printFullRow(X_trainCat)
    X_train, y_train, X_val, y_val, test_train, test_val = getAllCleanedData(standardize=1)
    # --- continuous dataset
    X_trainCon = X_train.drop(['NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Germany', 'Spain', 'Male'],axis=1)
    X_ValCon = X_val.drop(['NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Germany', 'Spain', 'Male'],axis=1)
    test_trainCon = test_train.drop(['NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Germany', 'Spain', 'Male'],axis=1)
    # printFullRow(X_trainCon)

    # --- XGBoost
    model = xgb.XGBClassifier()
    params = {
        'n_estimators': [15, 20],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
        'max_depth': [4, 5, 6],
        'gamma': [0, 0.1, 0.3, 0.5, 0.7, 1],
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "min_child_weight": [1, 3, 5, 7],
        'validate_parameters': [0]
    }

    XGB = RandomizedSearchCV(model, params, verbose=1, n_jobs=5, cv=3, scoring='f1')
    XGB.fit(X_trainCat, y_train)
    XGB_pred_prob = XGB.predict_proba(test_trainCat)
    XGB_pred_1 = XGB_pred_prob[:,0]
    # print(XGB_pred)

    # --- keras
    keras = kerasModel()
    early_stopping = EarlyStopping(
        monitor='get_f1',
        verbose=1,
        patience=20,
        mode='min',
        restore_best_weights=True)

    keras.fit(
        X_trainCon,
        y_train,
        batch_size=3,
        epochs=10,
        callbacks=[early_stopping]
    )
    keras_pred_prob = keras.predict(test_trainCon)
    # probability of being class 1
    keras_pred_1 = keras_pred_prob[:,0]

    y_pred = (keras_pred_prob > 0.5)
    # cm = confusion_matrix(test_val, y_pred)
    # print(cm)
    # print("Accuracy score: ", accuracy_score(test_val, y_pred))
    # print("Precision score: ", precision_score(test_val, y_pred))
    # print("Recall score: ", recall_score(test_val, y_pred))
    print("F1 score: ", f1_score(test_val, y_pred))

    threshold = 0.65
    combined_pred_prob = []
    combined_pred = []
    XGB_pred_1 = XGB_pred_1.tolist()
    keras_pred_1 = keras_pred_1.tolist()
    # stack both models
    for x in range(len(XGB_pred_1)):
        if XGB_pred_1[x] >= threshold or keras_pred_1[x] >= threshold:
            combined_pred.append(1)
        else:
            combined_pred.append(0)
    # for x in combined_pred_prob:
    #     if x >0.5:
    #         combined_pred.append(1)
    #     else:
    #         combined_pred.append(0)

    cm = confusion_matrix(test_val, combined_pred)
    print(cm)
    print("Accuracy score: ", accuracy_score(test_val, combined_pred))
    print("Precision score: ", precision_score(test_val, combined_pred))
    print("Recall score: ", recall_score(test_val, combined_pred))
    print("F1 score: ", f1_score(test_val, combined_pred))


    return

customEnsemble()