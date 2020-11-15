
from sklearn.decomposition import PCA
from tensorflow._api.v2 import train
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer
import pandas as pd
from FTEC4003.utils import printFullRow


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
        printFullRow(result)
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
    return df

def standard(X_train, X_test):
    # Standardization, or mean removal and variance scaling
    scale = StandardScaler().fit(X_train)
    X_train = scale.transform(X_train)
    X_test = scale.transform(X_test)
    return X_train, X_test

def scaling(df):
    # Scaling features to a range
    min_max_scaler = MinMaxScaler().fit(df)
    df = min_max_scaler.transform(df)
    return df

def normalization(df):
    # Normalization is the process of scaling individual samples to have unit norm
    # The L2-norm is the usual Euclidean length, i.e. the square root of the sum of the squared vector elements.
    # The L1-norm is the sum of the absolute values of the vector elements.
    # Default setting is L2-norm
    normalizer = Normalizer().fit(df)
    df = normalizer.transform(df)
    return df

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

    # ----- visualize data -----
    #if visualize:
         #----- description -----
        #description(train)
        # ----- histogram -----
        #histogram(train)
        # ----- correlation analysis -----
        #corrAnalysis(train)
        corrAnalysis(train)
        print(len(train[train['Exited'] == 1]))
    # return train.drop(columns=['Spain', 'Female', 'NotActive'])
    return train


def simpleGetData(filename):
    train = pd.read_csv(filename, header=0)
    train.drop(['RowNumber', 'Surname', 'CustomerId'], axis=1, inplace=True)
    return train