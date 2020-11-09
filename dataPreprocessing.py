import category_encoders as ce
from utils import *
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer

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
        result = pd.concat([df, enc_df], axis=1)
        printFullRow(result)
    else:
        print('process task 1')
    return result


def binaryEncoding(df):
    # binaryEncoding is better than one-hot Encoding for features that have many values
    encoder = ce.BinaryEncoder(cols=['Geography'])
    df_binary = encoder.fit_transform(df)
    return df_binary

def discretization(df):

    return df

def standard(df):
    # Standardization, or mean removal and variance scaling
    scale = StandardScaler().fit(df)
    df = scale.transform(df)
    return df

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

def getTrainingData(filename, visualize=False):
    # ----- loading data -----
    train = pd.read_csv(filename, header=0)
    # Task 1
    if 'insurance' in filename:
        print("preprocessing insurance-train.csv")
    # Task 2
    else:
        # drop the RowNumber, CustomerId and Surname as they are not important
        train.drop(['RowNumber', 'Surname', 'CustomerId'], axis=1, inplace=True)
        train = oneHotEncoding(train, 2)

    # ----- visualize data -----
    if visualize:
        # ----- description -----
        description(train)
        # ----- histogram -----
        histogram(train)
        # ----- correlation analysis -----
        corrAnalysis(train)
    return train

def simpleGetData(filename):
    train = pd.read_csv(filename, header=0)
    train.drop(['RowNumber', 'Surname', 'CustomerId'], axis=1, inplace=True)
    return train