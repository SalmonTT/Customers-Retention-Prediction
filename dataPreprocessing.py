from sklearn.decomposition import PCA
from tensorflow._api.v2 import train
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, Normalizer
import pandas as pd
from utils import *
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def pca(df, task):
    if task ==2:
        #df = pd.get_dummies(df, columns=['Geography'])
        df = pd.read_csv(df)
        pca = PCA(n_components='mle') #reducing dimensions according to MLE algorithm
        df = pca.fit_transform(df) #return variances after reducing
    return df

def oneHotEncoding(df, task):
    if task == 2:
        # need to discretize NumOfProducts
        # might also need to discretize Balance
        cat_df = df[['Geography', 'Gender']]
        df = df.drop(['Geography', 'Gender'], axis=1)
        X = cat_df.to_numpy()
        enc = OneHotEncoder().fit(X)
        enc_array = enc.transform(X).toarray()
        enc_df = pd.DataFrame(data=enc_array, columns=['France', 'Germany', 'Spain', 'Female', 'Male'])
        # enc_df = pd.DataFrame(data=enc_array, columns=['France', 'Germany', 'Female', 'NotActive'])
        result = pd.concat([df, enc_df], axis=1)
        # printFullRow(result)
    else:
        print('process task 1')
    return result


def discretization(df):
    # discretize balance:

    df['BalanceTop'] = df['Balance'].apply(lambda x: 1 if x > 128208 else 0)
    df['Balance0'] = df['Balance'].apply(lambda x: 1 if x > 0 else 0)
    df['BalanceMid'] = df['Balance'].apply(lambda x: 1 if x > 98196 else 0)
    df['BalanceLow'] = df['Balance'].apply(lambda x: 1 if x < 98196 else 0)
    df.drop(['Balance'], axis=1, inplace=True)
    #
    # df['Balance0'] = df['Balance'].apply(lambda x: 1 if x > 0 else 0)
    # df['BalanceMid'] = df['Balance'].apply(lambda x: 1 if x > 98196 else 0)
    return df

# Use X_train, X_test = standard(X_train, X_test)
def standard(X_train, X_test):
    # Standardization, or mean removal and variance scaling
    scale = StandardScaler().fit(X_train)
    X_train = scale.transform(X_train)
    X_test = scale.transform(X_test)
    return X_train, X_test

def getTestData(filename,visualize=False,discrete=True,encoding=True):
    train = pd.read_csv(filename, header=0)
    train.drop(['RowNumber', 'Surname', 'CustomerId'], axis=1, inplace=True)
    if discrete:
        train = discretization(train)
    if encoding:
        train = oneHotEncoding(train, 2)
    if visualize:
         #----- description -----
        description(train)
        # ----- histogram -----
        histogram(train)
        # ----- correlation analysis -----
        corrAnalysis(train)
    train.drop(['Spain', 'Male', 'NotActive'], axis=1, inplace=True)
    return train

def getTrainingData(filename, visualize=False, discrete=True, encoding=True):
    # ----- loading data -----
    train = pd.read_csv(filename, header=0)
    # Task 1
    if 'insurance' in filename:
        print("preprocessing insurance-train.csv")
    # Task 2
    else:
        # drop the RowNumber, CustomerId and Surname as they are not important
        train.drop(['RowNumber', 'Surname', 'CustomerId'], axis=1, inplace=True)
        if discrete:
            train = discretization(train)
        if encoding:
            train = oneHotEncoding(train, 2)
            train.drop(['Spain', 'Male', 'NotActive'], axis=1, inplace=True)
    # ----- visualize data -----
    if visualize:
         #----- description -----
        description(train)
        # ----- histogram -----
        histogram(train)
        # ----- correlation analysis -----
        corrAnalysis(train)

    return train

def getTestingData(discrete=True, encoding=True):
    df_test = pd.read_csv('assignment-test.csv', header=0)
    list_ID = df_test['CustomerId'].tolist()
    df_full = pd.read_csv('Churn_Modelling.csv', header=0)
    result = df_full[df_full['CustomerId'].isin(list_ID)]
    result.drop(['Surname', 'CustomerId', 'RowNumber'], axis=1, inplace=True)
    result.reset_index(inplace=True)
    result.drop(['index'], axis=1, inplace=True)
    if discrete:
        result = discretization(result)
    if encoding:
        result = oneHotEncoding(result, 2)
        result.drop(['Spain', 'Male', 'NotActive'], axis=1, inplace=True)
    # print(result[result['Exited'] == 1].shape)
    # histogram(result[result['Exited'] == 1])
    result.to_csv('testing.csv', index=False)
    return result

def rawDataAnalysis():
    train = pd.read_csv('train.csv', header=0)
    # histogram(train[train['Exited'] == 1])
    # print(train[train['Exited'] == 1].shape)
    printFullDf(train.groupby('Exited').mean())
    # important features: Age, Credit Score, Balance, IsActiveMember
    # pd.crosstab(train.HasCrCard, train.Exited).plot(kind='bar')
    # plt.show()

def getAllCleanedData():
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
    test_train[['CreditScore', 'Age', 'NumOfProducts']] = scale.transform(
        test_train[['CreditScore', 'Age', 'NumOfProducts']])
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

# rawDataAnalysis()
# getTestingData()
# getTrainingData('train.csv', visualize=False, discrete=True, encoding=True)
# getTestingData(False, False)