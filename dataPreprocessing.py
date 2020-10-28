import pandas as pd
import category_encoders as ce

def oneHotEncoding(df):
    df = pd.get_dummies(df, columns=['Geography'])
    return df

def binaryEncoding(df):
    # binaryEncoding is better than one-hot Encoding for features that have many values
    encoder = ce.BinaryEncoder(cols=['Geography'])
    df_binary = encoder.fit_transform(df)
    return df_binary

def getTrainingData():
    # ----- loading data -----
    # train: 7500*13
    train = pd.read_csv('train.csv', header=0)
    # drop the RowNumber and Surname as they are not important, we keep 'CustomerId' for identification later
    train.drop(['RowNumber', 'Surname'], axis=1, inplace=True)
    # -----Visualizing the distribution of the data for every feature -----
    # train.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20))
    # plt.show()

    # --- Observations ---:
    # 'CreditScore' follows a Gausian distribution (relatively)
    # 'Age' follows Gausian distribution skewed to the left
    # 'Balance' has extreme outliers
    # most people have less than 3 products
    # 'EstimateSalary' has low variance, may not be very informative
    # 'IsActiveMember' has high entropy: about half the people are active
    # 'HasCrCard' has relatively low entropy: about 2/3 have credit card

    # ----- feature engineering -----
    # change 'Gender' into numeric value
    train['Gender'].replace({'Male': 0, 'Female': 1}, inplace=True)
    # using the onehot method to label 'Geography'
    train_onehot = oneHotEncoding(train)
    # printFullRow(train_onehot)
    # using the binaryEncoding method to label 'Geography'
    train_binary_encoding = binaryEncoding(train)
    # printFullRow(train_binary_encoding)

    # ----- correlation test (Pearson) -----
    # corrTest(train_onehot)

    return train_onehot