import pandas as pd
import category_encoders as ce
from utils import *
import matplotlib as plt

def oneHotEncoding(df):
    df = pd.get_dummies(df, columns=['Geography'])
    return df


def binaryEncoding(df):
    # binaryEncoding is better than one-hot Encoding for features that have many values
    encoder = ce.BinaryEncoder(cols=['Geography'])
    df_binary = encoder.fit_transform(df)
    return df_binary


def getTrainingData(filename, visualize=False):
    # ----- loading data -----
    train = pd.read_csv(filename, header=0)
    # Task 2
    if filename == 'train.csv':
        # drop the RowNumber, CustomerId and Surname as they are not important
        train.drop(['RowNumber', 'Surname', 'CustomerId'], axis=1, inplace=True)
        train['Gender'].replace({'Male': 0, 'Female': 1}, inplace=True)
        train = oneHotEncoding(train)

    # Task 1
    else:
        print("preprocessing insurance-train.csv")

    # ----- visualize data -----
    if visualize:
        # ----- description -----
        # description(train)
        # ----- histogram -----
        # histogram(train)
        # ----- correlation analysis -----
        corrAnalysis(train)
    return train
