from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from utils import *
from dataPreprocessing import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
from sklearn import model_selection
from sklearn.metrics import classification_report


def logis():
    df = pd.read_csv('train.csv', header=0)
    # printFullRow(df)
    # print(df['Exited'].value_counts())
    df.drop(['CustomerId', 'Surname', 'RowNumber'], axis=1, inplace=True)
    X = df.drop(['Exited'], axis=1)
    y = df['Exited']
    # printFullRow(X_train.head())

    df_test = pd.read_csv('testing.csv', header=0)
    # print(df_test.info())
    test_train = df_test.drop(['Exited'],axis=1)
    test_val = df_test['Exited']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=1)
    print(len(y_train), len(y_val))

    ##### ENCODING #####

    X_train['HasCrCard'] = X_train['HasCrCard'].apply(lambda x: 1. if x == 1 else 0.)
    X_val['HasCrCard'] = X_val['HasCrCard'].apply(lambda x: 1. if x == 1 else 0.)
    test_train['HasCrCard'] = test_train['HasCrCard'].apply(lambda x: 1. if x == 1 else 0.)

    X_train['IsActiveMember'] = X_train['IsActiveMember'].apply(lambda x: 1. if x == 1 else 0.)
    X_val['IsActiveMember'] = X_val['IsActiveMember'].apply(lambda x: 1. if x == 1 else 0.)
    test_train['IsActiveMember'] = test_train['IsActiveMember'].apply(lambda x: 1. if x == 1 else 0.)

    X_train_cat_df = X_train[['Geography', 'Gender']]
    X_val_cat_df = X_val[['Geography', 'Gender']]
    test_cat_df = test_train[['Geography', 'Gender']]

    X_train = X_train.drop(['Geography', 'Gender'], axis=1)
    X_val = X_val.drop(['Geography', 'Gender'], axis=1)
    test_train = test_train.drop(['Geography', 'Gender'], axis=1)
    X_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    test_train.reset_index(drop=True, inplace=True)

    X_train_cat = X_train_cat_df.to_numpy()
    X_val_cat = X_val_cat_df.to_numpy()
    test_cat = test_cat_df.to_numpy()

    enc = OneHotEncoder().fit(X_train_cat)
    X_train_enc_array = enc.transform(X_train_cat).toarray()
    X_val_enc_array = enc.transform(X_val_cat).toarray()
    test_enc_array = enc.transform(test_cat).toarray()

    X_train_enc_df = pd.DataFrame(data=X_train_enc_array, columns=['France', 'Germany', 'Spain', 'Female', 'Male'])
    X_val_enc_df = pd.DataFrame(data=X_val_enc_array, columns=['France', 'Germany', 'Spain', 'Female', 'Male'])
    test_enc_df = pd.DataFrame(data=test_enc_array, columns=['France', 'Germany', 'Spain', 'Female', 'Male'])

    X_train = pd.concat([X_train, X_train_enc_df], axis=1)
    X_val = pd.concat([X_val, X_val_enc_df], axis=1)
    test_train = pd.concat([test_train, test_enc_df], axis=1)

    # drop the extra columns
    X_train.drop(['France', 'Female'], axis=1, inplace=True)
    X_val.drop(['France', 'Female'], axis=1, inplace=True)
    test_train.drop(['France', 'Female'], axis=1, inplace=True)


    ###### Oversample training  data #####
    smk = SMOTE()
    X_train, y_train = smk.fit_sample(X_train, y_train)
    # Oversample validation data
    X_val, y_val = smk.fit_sample(X_val, y_val)

    # print(X_train.shape, X_val.shape)
    # printFullRow(X_train[:5])
    # print(y_train.value_counts())

    ###### Standarize and Normalize #####

    scale = StandardScaler().fit(X_train[['CreditScore', 'Age', 'NumOfProducts']])
    X_train[['CreditScore', 'Age', 'NumOfProducts']] = scale.transform(X_train[['CreditScore', 'Age', 'NumOfProducts']])
    X_val[['CreditScore', 'Age', 'NumOfProducts']] = scale.transform(X_val[['CreditScore', 'Age', 'NumOfProducts']])
    test_train[['CreditScore', 'Age', 'NumOfProducts']] = scale.transform(test_train[['CreditScore', 'Age', 'NumOfProducts']])
    robust_scale = RobustScaler().fit(X_train[['Balance']])
    X_train[['Balance']] = robust_scale.transform(X_train[['Balance']])
    X_val[['Balance']] = robust_scale.transform(X_val[['Balance']])
    test_train[['Balance']] = robust_scale.transform(test_train[['Balance']])
    # normalize salary
    salary_mean = X_train['EstimatedSalary'].mean()
    salary_std = X_train['EstimatedSalary'].std()
    X_train['EstimatedSalary'] = X_train['EstimatedSalary'].apply(lambda x: (x - salary_mean)/salary_std)
    salary_mean = X_val['EstimatedSalary'].mean()
    salary_std = X_val['EstimatedSalary'].std()
    X_val['EstimatedSalary'] = X_val['EstimatedSalary'].apply(lambda x: (x - salary_mean) / salary_std)
    salary_mean = test_train['EstimatedSalary'].mean()
    salary_std = test_train['EstimatedSalary'].std()
    test_train['EstimatedSalary'] = test_train['EstimatedSalary'].apply(lambda x: (x - salary_mean) / salary_std)
    # normalize tenure
    tenure_mean = X_train['Tenure'].mean()
    tenure_std = X_train['Tenure'].std()
    X_train['Tenure'] = X_train['Tenure'].apply(lambda x: (x - tenure_mean) / tenure_std)
    tenure_mean = X_val['Tenure'].mean()
    tenure_std = X_val['Tenure'].std()
    X_val['Tenure'] = X_val['Tenure'].apply(lambda x: (x - tenure_mean) / tenure_std)
    tenure_mean = test_train['Tenure'].mean()
    tenure_std = test_train['Tenure'].std()
    test_train['Tenure'] = test_train['Tenure'].apply(lambda x: (x - tenure_mean) / tenure_std)

    printFullRow(X_train.head())
    print(description(X_train))

    ##### MODEL BUILDING ######
    f1_scores = []
    models = []
    kfold = 3
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    # --- Logistic regression
    logistic_model = LogisticRegression(solver='lbfgs', max_iter=300)
    # fit the data
    logistic_model.fit(X_train, y_train)
    print("using train.csv")
    print_score(logistic_model, X_train, y_train, X_val, y_val, train=True)
    print_score(logistic_model, X_train, y_train, X_val, y_val, train=False)
    print('using testing.csv')
    score = print_score(logistic_model, X_train, y_train, test_train, test_val, train=False)
    f1_scores.append(score)
    models.append('logistics regression')

    # --- Logistic regression

    return

logis()