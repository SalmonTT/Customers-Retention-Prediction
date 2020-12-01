from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from utils import printFullDf
from imblearn.over_sampling import SVMSMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, Normalizer
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from dataPreprocessing import getAllCleanedData

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    f1 = 0
    if train:
        pred = clf.predict(X_train)
        print("Train Result:\n===========================================")
        print(f"accuracy score: {accuracy_score(y_train, pred):.4f}\n")
        print(
            f"Classification Report: \n \tPrecision: {precision_score(y_train, pred)}\n\tRecall Score: {recall_score(y_train, pred)}\n\tF1 score: {f1_score(y_train, pred)}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, clf.predict(X_train))}\n")

    elif train == False:
        pred = clf.predict(X_test)
        print("Test Result:\n===========================================")
        print(f"accuracy score: {accuracy_score(y_test, pred)}\n")
        print(
            f"Classification Report: \n \tPrecision: {precision_score(y_test, pred)}\n\tRecall Score: {recall_score(y_test, pred)}\n\tF1 score: {f1_score(y_test, pred)}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
        f1 = f1_score(y_test, pred)
    return f1

def getAllCleanedDataExperiment(binning=0):
    # https://www.kaggle.com/nasirislamsujan/bank-customer-churn-prediction for inspiration on feature engineering
    # https://www.kaggle.com/agustinpugliese/eda-86-ann-explained
    # https://www.kaggle.com/kmalit/bank-customer-churn-prediction
    # https://www.kaggle.com/nasirislamsujan/bank-customer-churn-prediction
    df = pd.read_csv('train.csv', header=0)
    # printFullRow(df)
    # print(df['Exited'].value_counts())
    df.drop(['CustomerId', 'Surname', 'RowNumber'], axis=1, inplace=True)
    X = df.drop(['Exited'], axis=1)
    y = df['Exited']
    # printFullRow(X_train.head())

    df_test = pd.read_csv('testing.csv', header=0)
    # print(df_test.info())
    test_train = df_test.drop(['Exited'], axis=1)
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
    svmsmote = SVMSMOTE(random_state=101)
    X_train, y_train = svmsmote.fit_resample(X_train, y_train)

    # binning num of products
    X_train['1 Product'] = X_train['NumOfProducts'].apply(lambda x: 1 if x==1  else 0)
    X_train['2 Product'] = X_train['NumOfProducts'].apply(lambda x: 1 if x==2 else 0)
    X_train['1/2 Product'] = X_train['NumOfProducts'].apply(lambda x: 1 if x <=2 else 0)
    X_train['3/4 Product'] = X_train['NumOfProducts'].apply(lambda x: 1 if x>=3 else 0)
    X_train.drop(['NumOfProducts'], axis=1, inplace=True)
    X_val['1 Product'] = X_val['NumOfProducts'].apply(lambda x: 1 if x == 1 else 0)
    X_val['2 Product'] = X_val['NumOfProducts'].apply(lambda x: 1 if x == 2 else 0)
    X_val['1/2 Product'] = X_val['NumOfProducts'].apply(lambda x: 1 if x <= 2 else 0)
    X_val['3/4 Product'] = X_val['NumOfProducts'].apply(lambda x: 1 if x >= 3 else 0)
    X_val.drop(['NumOfProducts'], axis=1, inplace=True)
    test_train['1 Product'] = test_train['NumOfProducts'].apply(lambda x: 1 if x == 1 else 0)
    test_train['2 Product'] = test_train['NumOfProducts'].apply(lambda x: 1 if x == 2 else 0)
    test_train['1/2 Product'] = test_train['NumOfProducts'].apply(lambda x: 1 if x <= 2 else 0)
    test_train['3/4 Product'] = test_train['NumOfProducts'].apply(lambda x: 1 if x >= 3 else 0)
    test_train.drop(['NumOfProducts'], axis=1, inplace=True)
    # X_train['Balance0'] = X_train['Balance'].apply(lambda x: 1 if x < 50000 else 0)
    # X_train['Balance1'] = X_train['Balance'].apply(lambda x: 1 if (x > 50000 and x < 100000) else 0)
    # X_train['Balance2'] = X_train['Balance'].apply(lambda x: 1 if (x > 100000 and x < 150000) else 0)
    # X_train['Balance3'] = X_train['Balance'].apply(lambda x: 1 if (x > 150000 and x < 200000) else 0)
    # X_train.drop(['Balance'], axis=1, inplace=True)
    # X_val['Balance0'] = X_val['Balance'].apply(lambda x: 1 if x < 50000 else 0)
    # X_val['Balance1'] = X_val['Balance'].apply(lambda x: 1 if (x > 50000 and x < 100000) else 0)
    # X_val['Balance2'] = X_val['Balance'].apply(lambda x: 1 if (x > 100000 and x < 150000) else 0)
    # X_val['Balance3'] = X_val['Balance'].apply(lambda x: 1 if (x > 150000 and x < 200000) else 0)
    # X_val.drop(['Balance'], axis=1, inplace=True)
    # test_train['Balance0'] = test_train['Balance'].apply(lambda x: 1 if x < 50000 else 0)
    # test_train['Balance1'] = test_train['Balance'].apply(lambda x: 1 if (x > 50000 and x < 100000) else 0)
    # test_train['Balance2'] = test_train['Balance'].apply(lambda x: 1 if (x > 100000 and x < 150000) else 0)
    # test_train['Balance3'] = test_train['Balance'].apply(lambda x: 1 if (x > 150000 and x < 200000) else 0)
    # test_train.drop(['Balance'], axis=1, inplace=True)

    # age
    X_train['Age40-70'] = X_train['Age'].apply(lambda x: 1 if (x>=40 and x<=70) else 0)
    X_val['Age40-70'] = X_val['Age'].apply(lambda x: 1 if (x >= 40 and x <= 70) else 0)
    test_train['Age40-70'] = test_train['Age'].apply(lambda x: 1 if (x >= 40 and x <= 70) else 0)
    # balance
    X_train['Balance-mid'] = X_train['Balance'].apply(lambda x: 1 if (x>=75000 and x<=160000) else 0)
    X_val['Balance-mid'] = X_val['Balance'].apply(lambda x: 1 if (x>=75000 and x<=160000) else 0)
    test_train['Balance-mid'] = test_train['Balance'].apply(lambda x: 1 if (x>=75000 and x<=160000) else 0)
    X_train['Balance-low'] = X_train['Balance'].apply(lambda x: 1 if (x <= 25000) else 0)
    X_val['Balance-low'] = X_val['Balance'].apply(lambda x: 1 if (x <= 25000) else 0)
    test_train['Balance-low'] = test_train['Balance'].apply(lambda x: 1 if (x <= 25000) else 0)
    # credit score
    X_train['LowCredit'] = X_train['CreditScore'].apply(lambda x: 1 if (x <= 400) else 0)
    X_val['LowCredit'] = X_val['CreditScore'].apply(lambda x: 1 if (x <= 400) else 0)
    test_train['LowCredit'] = test_train['CreditScore'].apply(lambda x: 1 if (x <= 400) else 0)
    X_train['HighCredit'] = X_train['CreditScore'].apply(lambda x: 1 if (x >= 800) else 0)
    X_val['HighCredit'] = X_val['CreditScore'].apply(lambda x: 1 if (x >= 800) else 0)
    test_train['HighCredit'] = test_train['CreditScore'].apply(lambda x: 1 if (x >= 800) else 0)
    # # tenure
    X_train['Tenure6-8'] = X_train['Tenure'].apply(lambda x: 1 if (x >= 6 and x <= 8) else 0)
    X_val['Tenure6-8'] = X_val['Tenure'].apply(lambda x: 1 if (x >= 6 and x <= 8) else 0)
    test_train['Tenure6-8'] = test_train['Tenure'].apply(lambda x: 1 if (x >= 6 and x <= 8) else 0)
    # X_train['Tenure1-2'] = X_train['Tenure'].apply(lambda x: 1 if (x >= 1 and x <= 2) else 0)
    # X_val['Tenure1-2'] = X_val['Tenure'].apply(lambda x: 1 if (x >= 1 and x <= 2) else 0)
    # test_train['Tenure1-2'] = test_train['Tenure'].apply(lambda x: 1 if (x >= 1 and x <= 2) else 0)


    # Age vs. Balance and CreditScore
    # X_train['Balance/Age'] = X_train['Balance']/X_train['Age']
    # X_val['Balance/Age'] = X_val['Balance'] / X_val['Age']
    # test_train['Balance/Age'] = test_train['CreditScore'] / test_train['Age']
    # X_train['CreditScore/Age'] = X_train['CreditScore'] / X_train['Age']
    # X_val['CreditScore/Age'] = X_val['CreditScore'] / X_val['Age']
    # test_train['CreditScore/Age'] = test_train['CreditScore'] / test_train['Age']

    X_train.drop(['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'Tenure'], axis=1, inplace=True)
    X_val.drop(['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'Tenure'], axis=1, inplace=True)
    test_train.drop(['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'Tenure'], axis=1, inplace=True)

    return X_train, y_train, X_val, y_val, test_train, test_val

def getAllCleanedDataStable(standardize = 0, binning=0):
    df = pd.read_csv('train.csv', header=0)
    # printFullRow(df)
    # print(df['Exited'].value_counts())
    df.drop(['CustomerId', 'Surname', 'RowNumber'], axis=1, inplace=True)
    X = df.drop(['Exited'], axis=1)
    y = df['Exited']
    # printFullRow(X_train.head())

    df_test = pd.read_csv('testing.csv', header=0)
    # print(df_test.info())
    test_train = df_test.drop(['Exited'], axis=1)
    test_val = df_test['Exited']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
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
    # random_state=101 gives 0.618
    # random_state=5 gives 0.61846
    svmsmote = SVMSMOTE(random_state=5)
    X_train, y_train = svmsmote.fit_resample(X_train, y_train)
    # smk = SMOTE()
    # X_train, y_train = smk.fit_sample(X_train, y_train)
    # adasyn = ADASYN(random_state=101)
    # X_train, y_train = adasyn.fit_resample(X_train, y_train)
    # over = RandomOverSampler(sampling_strategy=0.4, random_state=42)
    # under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    # X_train, y_train = over.fit_sample(X_train, y_train)
    # X_train, y_train = under.fit_sample(X_train, y_train)

    # print(X_train.shape, X_val.shape)
    # printFullRow(X_train[:5])
    # print(y_train.value_counts())

    if binning == 1:
        # bin balance into 4 categories
        # X_train['Balance0'] = X_train['Balance'].apply(lambda x: 1 if x < 50000 else 0)
        # X_train['Balance1'] = X_train['Balance'].apply(lambda x: 1 if (x > 50000 and x < 100000) else 0)
        # X_train['Balance2'] = X_train['Balance'].apply(lambda x: 1 if (x > 100000 and x < 150000) else 0)
        # X_train['Balance3'] = X_train['Balance'].apply(lambda x: 1 if (x > 150000 and x < 200000) else 0)
        # X_train.drop(['Balance'], axis=1, inplace=True)
        # X_val['Balance0'] = X_val['Balance'].apply(lambda x: 1 if x < 50000 else 0)
        # X_val['Balance1'] = X_val['Balance'].apply(lambda x: 1 if (x > 50000 and x < 100000) else 0)
        # X_val['Balance2'] = X_val['Balance'].apply(lambda x: 1 if (x > 100000 and x < 150000) else 0)
        # X_val['Balance3'] = X_val['Balance'].apply(lambda x: 1 if (x > 150000 and x < 200000) else 0)
        # X_val.drop(['Balance'], axis=1, inplace=True)
        # test_train['Balance0'] = test_train['Balance'].apply(lambda x: 1 if x < 50000 else 0)
        # test_train['Balance1'] = test_train['Balance'].apply(lambda x: 1 if (x > 50000 and x < 100000) else 0)
        # test_train['Balance2'] = test_train['Balance'].apply(lambda x: 1 if (x > 100000 and x < 150000) else 0)
        # test_train['Balance3'] = test_train['Balance'].apply(lambda x: 1 if (x > 150000 and x < 200000) else 0)
        # test_train.drop(['Balance'], axis=1, inplace=True)

        # binning age
        # X_train['Age40-50'] = X_train['Age'].apply(lambda x: 1 if (x>=40 and x<50) else 0)
        # X_train['Age30-40'] = X_train['Age'].apply(lambda x: 1 if (x >= 30 and x < 40) else 0)
        # X_train['Ageless30'] = X_train['Age'].apply(lambda x: 1 if (x < 30) else 0)
        # X_train['Ageover50'] = X_train['Age'].apply(lambda x: 1 if (x > 50) else 0)
        # X_train.drop(['Age'], axis=1, inplace=True)
        # X_val['Age40-50'] = X_val['Age'].apply(lambda x: 1 if (x >= 40 and x < 50) else 0)
        # X_val['Age30-40'] = X_val['Age'].apply(lambda x: 1 if (x >= 30 and x < 40) else 0)
        # X_val['Ageless30'] = X_val['Age'].apply(lambda x: 1 if (x < 30) else 0)
        # X_val['Ageover50'] = X_val['Age'].apply(lambda x: 1 if (x > 50) else 0)
        # X_val.drop(['Age'], axis=1, inplace=True)
        # test_train['Age40-50'] = test_train['Age'].apply(lambda x: 1 if (x>=40 and x<50) else 0)
        # test_train['Age30-40'] = test_train['Age'].apply(lambda x: 1 if (x >= 30 and x < 40) else 0)
        # test_train['Ageless30'] = test_train['Age'].apply(lambda x: 1 if (x < 30) else 0)
        # test_train['Ageover50'] = test_train['Age'].apply(lambda x: 1 if (x > 50) else 0)
        # test_train.drop(['Age'], axis=1, inplace=True)

        # binning num of products
        X_train['1 Product'] = X_train['NumOfProducts'].apply(lambda x: 1 if x==1  else 0)
        X_train['2 Product'] = X_train['NumOfProducts'].apply(lambda x: 1 if x==2 else 0)
        X_train['3 Product'] = X_train['NumOfProducts'].apply(lambda x: 1 if x==3 else 0)
        X_train['4 Product'] = X_train['NumOfProducts'].apply(lambda x: 1 if x==4 else 0)
        X_train.drop(['NumOfProducts'], axis=1, inplace=True)
        X_val['1 Product'] = X_val['NumOfProducts'].apply(lambda x: 1 if x == 1 else 0)
        X_val['2 Product'] = X_val['NumOfProducts'].apply(lambda x: 1 if x == 2 else 0)
        X_val['3 Product'] = X_val['NumOfProducts'].apply(lambda x: 1 if x == 3 else 0)
        X_val['4 Product'] = X_val['NumOfProducts'].apply(lambda x: 1 if x == 4 else 0)
        X_val.drop(['NumOfProducts'], axis=1, inplace=True)
        test_train['1 Product'] = test_train['NumOfProducts'].apply(lambda x: 1 if x == 1 else 0)
        test_train['2 Product'] = test_train['NumOfProducts'].apply(lambda x: 1 if x == 2 else 0)
        test_train['3 Product'] = test_train['NumOfProducts'].apply(lambda x: 1 if x == 3 else 0)
        test_train['4 Product'] = test_train['NumOfProducts'].apply(lambda x: 1 if x == 4 else 0)
        test_train.drop(['NumOfProducts'], axis=1, inplace=True)

    ##### Control
    if standardize != 1:
        X_train_og = X_train
        X_val_og = X_val
        y_train_og = y_train
        y_val_og = y_val
        test_train_og = test_train
        test_val_og = test_val
        return X_train_og, y_train_og, X_val_og, y_val_og, test_train_og, test_val_og

    ###### Standarize and Normalize #####

    # scale = StandardScaler().fit(X_train[['CreditScore', 'Age', 'NumOfProducts']])
    # X_train[['CreditScore', 'Age', 'NumOfProducts']] = scale.transform(X_train[['CreditScore', 'Age', 'NumOfProducts']])
    # X_val[['CreditScore', 'Age', 'NumOfProducts']] = scale.transform(X_val[['CreditScore', 'Age', 'NumOfProducts']])
    # test_train[['CreditScore', 'Age', 'NumOfProducts']] = scale.transform(
    #     test_train[['CreditScore', 'Age', 'NumOfProducts']])

    scale = StandardScaler().fit(X_train[['CreditScore', 'NumOfProducts']])
    X_train[['CreditScore', 'NumOfProducts']] = scale.transform(X_train[['CreditScore', 'NumOfProducts']])
    X_val[['CreditScore', 'NumOfProducts']] = scale.transform(X_val[['CreditScore', 'NumOfProducts']])
    test_train[['CreditScore', 'NumOfProducts']] = scale.transform(
        test_train[['CreditScore', 'NumOfProducts']])

    robust_scale = RobustScaler().fit(X_train[['Balance']])
    X_train[['Balance']] = robust_scale.transform(X_train[['Balance']])
    X_val[['Balance']] = robust_scale.transform(X_val[['Balance']])
    test_train[['Balance']] = robust_scale.transform(test_train[['Balance']])

    # normalize salary
    salary_mean = X_train['EstimatedSalary'].mean()
    salary_std = X_train['EstimatedSalary'].std()
    X_train['EstimatedSalary'] = X_train['EstimatedSalary'].apply(lambda x: (x - salary_mean) / salary_std)
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

    return X_train, y_train, X_val, y_val, test_train, test_val

def XGModelCV():
    X_train, y_train, X_val, y_val, test_train, test_val = getAllCleanedData(binning=1)
    printFullDf(X_train.head())

    model = xgb.XGBClassifier()
    params = {
        'n_estimators': [15, 20],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
        'max_depth': [4, 5, 6],
        'gamma': [0, 0.1, 0.3 , 0.5, 0.7, 1],
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
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

def XGModel618():
    X_train, y_train, X_val, y_val, test_train, test_val = getAllCleanedDataStable(binning=1)
    printFullDf(X_train.head())

    model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.2, max_delta_step=0, max_depth=6,
              min_child_weight=3, monotone_constraints='()',
              n_estimators=15, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=0, verbosity=None)
    model.fit(X_train, y_train)
    # importance = model.feature_importances_
    # # summarize feature importance
    # for i, v in enumerate(importance):
    #     print('Feature: %s, Score: %.5f' % (i, v))
    print_score(model, X_train, y_train, X_val, y_val, train=True)
    print_score(model, X_train, y_train, X_val, y_val, train=False)
    print_score(model, X_train, y_train, test_train, test_val, train=False)

def XGModelExperiment():
    X_train, y_train, X_val, y_val, test_train, test_val = getAllCleanedDataExperiment(binning=1)
    X_train_dummies = pd.get_dummies(X_train)
    X_val_dummies = pd.get_dummies(X_val)
    test_train_dummies = pd.get_dummies(test_train)

    printFullDf(X_train.head())
    # DMatrix
    train = xgb.DMatrix(data=X_train_dummies, label=y_train)
    val = xgb.DMatrix(data=X_val_dummies, label=y_val)
    test = xgb.DMatrix(data=test_train_dummies, label=test_val)

    params = {"base_score":0.5, "booster":'gbtree', "colsample_bylevel":1,
              "colsample_bynode":1, "colsample_bytree":0.4, "gamma":0.1, "gpu_id":-1,
              "importance_type":'gain', "interaction_constraints":'',
              "learning_rate":0.2, 'max_delta_step':0, 'max_depth':5,
              'min_child_weight':1, 'monotone_constraints':'()',
              'n_estimators':50, 'n_jobs':0, 'num_parallel_tree':1, 'random_state':0,
              'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1, 'subsample':1,
              'tree_method':'exact', 'validate_parameters':0, 'verbosity':None, 'objective':'binary:hinge'}
    params['eval_metric'] = 'auc'

    evallist = [(val, 'eval'), (train,'train')]
    num_rounds = 20
    bst = xgb.train(params, train, num_rounds, evallist, early_stopping_rounds=4)
    ypred = bst.predict(test, ntree_limit=bst.best_ntree_limit)
    print(f1_score(test_val, ypred))
    # print(bst.get_score(importance_type='gain'))
    # print(bst.get_score(importance_type='weight'))
    xgb.plot_importance(bst, importance_type='gain')
    xgb.plot_importance(bst, importance_type='weight')
    xgb.plot_tree(bst)
    plt.show()

XGModel618()
