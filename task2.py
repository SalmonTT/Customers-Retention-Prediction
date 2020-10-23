import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV


def printFullDf(df):
    # prints the dataframe in full
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)

def printFullRow(df):
    pd.set_option('display.max_columns', None)
    print(df)

def corrTest(df):
    # Part 1 -----------
    # Test the correlation between attributes/features (attributes and features will be used interchangeably):
    # drop the following column from dataframe
    feature = df.drop(['CustomerId', 'Exited'], axis=1)
    feature_corr = feature.corr(method='pearson')
    # print the correlation matrix
    printFullDf(feature_corr)
    # show the correlation heatmap
    plt.matshow(feature_corr)
    plt.show()
    # from the corr matrix we see that most features are uncorrelated (excluding geography),
    # with the exception of 'Balance' and 'NumOfProducts' --> Pearson value = -0.308252

    # Part 2 -----------
    # we test the correlation between features and output ('Exited')
    feature_output = df.drop(['CustomerId'], axis=1)
    feature_output_corr = feature_output.corr(method='pearson')
    output_corr = abs(feature_output_corr['Exited'])
    printFullDf(output_corr)
    # 'Age' show significant correlation to output 'Exited' --> Pearson value = 0.289895
    # 'IsActiveMember', 'Gender', 'Balance', 'Geography' are all moderately related
    return

def oneHotEncoding(df):
    df = pd.get_dummies(df, columns=['Geography'])
    return df

def binaryEncoding(df):
    # binaryEncoding is better than one-hot Encoding for features that have many values
    encoder = ce.BinaryEncoder(cols=['Geography'])
    df_binary = encoder.fit_transform(df)
    return df_binary

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
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

def decisionTreeTuned(X_train, X_test, y_train, y_test, df):
    # Tuning the hyper-parameters
    params = {
        "criterion": ("gini", "entropy"),
        "splitter": ("best", "random"),
        "max_depth": (list(range(1, 10))),
        "min_samples_split": list(range(10, 30)),
        "min_samples_leaf": list(range(1, 20)),
    }

    model = DecisionTreeClassifier(random_state=42)
    # n_jobs = -1: using all processors, cv=5: 5 cross-validation
    grid_search_cv = GridSearchCV(model, params, scoring="accuracy", verbose=1, n_jobs=-1, cv=3)
    grid_search_cv.fit(X_train, y_train)
    print(grid_search_cv.best_params_)
    # {'criterion': 'entropy', 'max_depth': 17, 'min_samples_leaf': 2, 'min_samples_split': 28, 'splitter': 'random'}

    # grid.search_cv.best_estimator_ is a DecisionTreeClassifier(max_depth=6, random_state=42, splitter='random')
    print_score(grid_search_cv.best_estimator_, X_train, y_train, X_test, y_test, train=True)
    print_score(grid_search_cv.best_estimator_, X_train, y_train, X_test, y_test, train=False)

    # visualizing tree
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 8), dpi=1600)
    plot_tree(grid_search_cv.best_estimator_, feature_names=df.columns.values, filled=True, class_names=['0', '1'])
    fig.savefig('decisionTree.png')

    return

def decisionTree(X_train, X_test, y_train, y_test, df):
    tree = DecisionTreeClassifier(criterion='gini',
                                  max_depth=6, max_features=None, max_leaf_nodes=None,
                                  min_impurity_decrease=0.0, min_impurity_split=None,
                                  min_samples_leaf=1, min_samples_split=2,
                                  random_state=42, splitter='random')
    tree.fit(X_train, y_train)

    print_score(tree, X_train, y_train, X_test, y_test, train=True)
    print_score(tree, X_train, y_train, X_test, y_test, train=False)

    # visualizing tree
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 8), dpi=1600)
    plot_tree(tree, feature_names=df.columns.values, filled=True, class_names=['0', '1'])
    fig.savefig('decisionTree.png')

def useDecisionTree(df):

    # ----- split training data -----
    X = df.drop(['Exited'], axis=1)
    y = df.Exited
    # test_size is the percentage of data allocated to test
    # random_state is a seed for random sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ----- decision tree classifier with tuned hyperparameters -----
    # note decisionTree() uses the tuned hyperparameters from decisionTreeTuned(), therefore
    # yields the same model
    decisionTreeTuned(X_train, X_test, y_train, y_test, X)

def dataPreprocessing():
    # ----- loading data -----
    # train: 7500*13
    train = pd.read_csv('train.csv', header=0)
    # drop the RowNumber and Surname as they are not important, we keep 'CustomerId' for identification later
    train.drop(['RowNumber', 'Surname'], axis=1, inplace=True)
    # -----Visualizing the distribution of the data for every feature -----
    train.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20))
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
    # print(train_onehot)
    # using the binaryEncoding method to label 'Geography'
    train_binary_encoding = binaryEncoding(train)
    # printFullRow(train_binary_encoding)

    # ----- correlation test (Pearson) -----
    # corrTest(train_onehot)

    return train_onehot

def main():
    # ----- Loading and Preprocessing data -----
    train = dataPreprocessing()

    # ----- Invoking models -----
    useDecisionTree(train)

if __name__ == "__main__":
    main()
