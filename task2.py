import sklearn
import pandas as pd
import matplotlib.pyplot as plt

def printDf(df):
    # prints the dataframe in full
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)

def corrTest(df):
    # Part 1 -----------
    # Test the correlation between attributes/features (attributes and features will be used interchangeably):
    # 'CreditScore' 'Geography' 'Gender' 'Age' 'Tenure' 'Balance' 'NumOfProducts' 'HasCrCard' 'IsActiveMember'
    # 'EstimatedSalary'

    # drop the following column from dataframe
    feature = df.drop(['CustomerId', 'Exited'], axis=1)
    feature_corr = feature.corr(method='pearson')
    # print the correlation matrix
    printDf(feature_corr)
    # show the correlation heatmap
    plt.matshow(feature_corr)
    plt.show()
    # from the corr matrix we see that most features are uncorrelated,
    # with the exception of 'Balance' and 'NumOfProducts' --> Pearson value = -0.308252

    # Part 2 -----------
    # we test the correlation between features and output ('Exited')
    feature_output = df.drop(['CustomerId'], axis=1)
    feature_output_corr = feature_output.corr(method='pearson')
    output_corr = abs(feature_output_corr['Exited'])
    printDf(output_corr)
    # 'Age' show significant correlation to output 'Exited' --> Pearson value = 0.289895
    # 'IsActiveMember' is moderately correlated to 'Exited' --> Pearson value = 0.159906
    # 'Balance' is moderately correlated to 'Exited' --> Pearson value = 0.127389
    return

def main():
    # df_test: 2500*13
    train = pd.read_csv('train.csv', header=0)
    # drop the RowNumber and Surname as they are not important, we keep 'CustomerId' for identification later
    train.drop(['RowNumber', 'Surname'], axis=1, inplace=True)
    # prints the column names of the training dataframe
    # print(train.columns.values)
    corrTest(train)


if __name__ == "__main__":
    main()