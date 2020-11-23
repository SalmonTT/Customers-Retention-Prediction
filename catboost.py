import catboost as cb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from utils import printFullDf
from imblearn.over_sampling import SVMSMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, Normalizer
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

def getAllCleanedData(binning=0):
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

    # # CAt encoding
    # X_train_cat_df = X_train[['Geography', 'Gender']]
    # X_val_cat_df = X_val[['Geography', 'Gender']]
    # test_cat_df = test_train[['Geography', 'Gender']]
    #
    # X_train = X_train.drop(['Geography', 'Gender'], axis=1)
    # X_val = X_val.drop(['Geography', 'Gender'], axis=1)
    # test_train = test_train.drop(['Geography', 'Gender'], axis=1)
    # X_train.reset_index(drop=True, inplace=True)
    # X_val.reset_index(drop=True, inplace=True)
    # test_train.reset_index(drop=True, inplace=True)
    #
    # X_train_cat = X_train_cat_df.to_numpy()
    # X_val_cat = X_val_cat_df.to_numpy()
    # test_cat = test_cat_df.to_numpy()
    #
    # enc = OneHotEncoder().fit(X_train_cat)
    # X_train_enc_array = enc.transform(X_train_cat).toarray()
    # X_val_enc_array = enc.transform(X_val_cat).toarray()
    # test_enc_array = enc.transform(test_cat).toarray()
    #
    # X_train_enc_df = pd.DataFrame(data=X_train_enc_array, columns=['France', 'Germany', 'Spain', 'Female', 'Male'])
    # X_val_enc_df = pd.DataFrame(data=X_val_enc_array, columns=['France', 'Germany', 'Spain', 'Female', 'Male'])
    # test_enc_df = pd.DataFrame(data=test_enc_array, columns=['France', 'Germany', 'Spain', 'Female', 'Male'])
    #
    # X_train = pd.concat([X_train, X_train_enc_df], axis=1)
    # X_val = pd.concat([X_val, X_val_enc_df], axis=1)
    # test_train = pd.concat([test_train, test_enc_df], axis=1)
    #
    # # drop the extra columns
    # X_train.drop(['France', 'Female'], axis=1, inplace=True)
    # X_val.drop(['France', 'Female'], axis=1, inplace=True)
    # test_train.drop(['France', 'Female'], axis=1, inplace=True)
    #
    # ###### Oversample training  data #####
    # svmsmote = SVMSMOTE(random_state=101)
    # X_train, y_train = svmsmote.fit_resample(X_train, y_train)

    # binning num of products
    # X_train['1 Product'] = X_train['NumOfProducts'].apply(lambda x: 1 if x==1  else 0)
    # X_train['2 Product'] = X_train['NumOfProducts'].apply(lambda x: 1 if x==2 else 0)
    # X_train['3 Product'] = X_train['NumOfProducts'].apply(lambda x: 1 if x==3 else 0)
    # X_train['4 Product'] = X_train['NumOfProducts'].apply(lambda x: 1 if x==4 else 0)
    # X_train.drop(['NumOfProducts'], axis=1, inplace=True)
    # X_val['1 Product'] = X_val['NumOfProducts'].apply(lambda x: 1 if x == 1 else 0)
    # X_val['2 Product'] = X_val['NumOfProducts'].apply(lambda x: 1 if x == 2 else 0)
    # X_val['3 Product'] = X_val['NumOfProducts'].apply(lambda x: 1 if x == 3 else 0)
    # X_val['4 Product'] = X_val['NumOfProducts'].apply(lambda x: 1 if x == 4 else 0)
    # X_val.drop(['NumOfProducts'], axis=1, inplace=True)
    # test_train['1 Product'] = test_train['NumOfProducts'].apply(lambda x: 1 if x == 1 else 0)
    # test_train['2 Product'] = test_train['NumOfProducts'].apply(lambda x: 1 if x == 2 else 0)
    # test_train['3 Product'] = test_train['NumOfProducts'].apply(lambda x: 1 if x == 3 else 0)
    # test_train['4 Product'] = test_train['NumOfProducts'].apply(lambda x: 1 if x == 4 else 0)
    # test_train.drop(['NumOfProducts'], axis=1, inplace=True)

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
    #
    # # binning age
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

    return X_train, y_train, X_val, y_val, test_train, test_val

def catboost():
    X_train, y_train, X_val, y_val, test_train, test_val = getAllCleanedData(binning=1)
    printFullDf(X_train.head())
    cat_features =
    params = {'loss_function':'Logloss', # objective function
          'eval_metric':'AUC', # metric
          'verbose': 200, # output to stdout info about training process every 200 iterations
          'random_seed': 42
         }





catboost()