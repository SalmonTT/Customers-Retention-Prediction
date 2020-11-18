from sklearn.decomposition import PCA
from tensorflow._api.v2 import train
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer
import pandas as pd
from utils import *


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
        cat_df = df[['Geography', 'Gender', 'IsActiveMember']]
        df = df.drop(['Geography', 'Gender', 'IsActiveMember'], axis=1)
        X = cat_df.to_numpy()
        enc = OneHotEncoder().fit(X)
        enc_array = enc.transform(X).toarray()
        enc_df = pd.DataFrame(data=enc_array, columns=['France', 'Germany', 'Spain', 'Female', 'Male', 'NotActive',
        'Active'])
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
    result.drop(['RowNumber', 'Surname', 'CustomerId'], axis=1, inplace=True)
    result.reset_index(inplace=True)
    result.drop(['index'], axis=1, inplace=True)
    if discrete:
        result = discretization(result)
    if encoding:
        result = oneHotEncoding(result, 2)
        result.drop(['Spain', 'Male', 'NotActive'], axis=1, inplace=True)
    # print(result[result['Exited'] == 1].shape)
    # histogram(result[result['Exited'] == 1])
    return result

def rawDataAnalysis():
    train = pd.read_csv('train.csv', header=0)
    # histogram(train[train['Exited'] == 1])
    # print(train[train['Exited'] == 1].shape)
    printFullDf(train.groupby('Exited').mean())
    # important features: Age, Credit Score, Balance, IsActiveMember
    # pd.crosstab(train.HasCrCard, train.Exited).plot(kind='bar')
    # plt.show()


rawDataAnalysis()
# getTestingData()
# getTrainingData('train.csv', visualize=False, discrete=True, encoding=True)