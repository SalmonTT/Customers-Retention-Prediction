from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import graphviz
from sklearn.model_selection import GridSearchCV, train_test_split
from utils import print_score, ROC
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from utils import printFullDf
from imblearn.over_sampling import SVMSMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, Normalizer

import pandas as pd


def getAllCleanedData(binning=0):
    # https://www.kaggle.com/nasirislamsujan/bank-customer-churn-prediction for inspiration on feature engineering
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
    X_train['3/4 Product'] = X_train['NumOfProducts'].apply(lambda x: 1 if x>=3 else 0)
    X_train.drop(['NumOfProducts'], axis=1, inplace=True)
    X_val['1 Product'] = X_val['NumOfProducts'].apply(lambda x: 1 if x == 1 else 0)
    X_val['2 Product'] = X_val['NumOfProducts'].apply(lambda x: 1 if x == 2 else 0)
    X_val['3/4 Product'] = X_val['NumOfProducts'].apply(lambda x: 1 if x >= 3 else 0)
    X_val.drop(['NumOfProducts'], axis=1, inplace=True)
    test_train['1 Product'] = test_train['NumOfProducts'].apply(lambda x: 1 if x == 1 else 0)
    test_train['2 Product'] = test_train['NumOfProducts'].apply(lambda x: 1 if x == 2 else 0)
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
    X_train['Balance-mid'] = X_train['Balance'].apply(lambda x: 1 if (x>=75000 and x<=16000) else 0)
    X_val['Balance-mid'] = X_val['Balance'].apply(lambda x: 1 if (x>=75000 and x<=16000) else 0)
    test_train['Balance-mid'] = test_train['Balance'].apply(lambda x: 1 if (x>=75000 and x<=16000) else 0)
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

def decisionTree():
    X_train, y_train, X_val, y_val, test_train, test_val = getAllCleanedData(binning=1)
    X_train_dummies = pd.get_dummies(X_train)
    X_val_dummies = pd.get_dummies(X_val)
    test_train_dummies = pd.get_dummies(test_train)

    printFullDf(X_train_dummies.head())
    # this always uses the best parameters
    tree = DecisionTreeClassifier(criterion='gini',
                                  max_depth=5, max_features=None, max_leaf_nodes=None,
                                  min_impurity_decrease=0.0, min_impurity_split=None,
                                  min_samples_leaf=1, min_samples_split=2,
                                  random_state=42, splitter='best')
    tree.fit(X_train_dummies, y_train)
    print_score(tree, X_train_dummies, y_train, X_val, y_val, train=True)
    print_score(tree, X_train_dummies, y_train, X_val_dummies, y_val, train=False)
    print_score(tree, X_train_dummies, y_train, test_train_dummies, test_val, train=False)
    visualizeTree(tree, X_train_dummies)
    return tree

def visualizeTree(tree, df):
    export_graphviz(tree, out_file='tree.dot',
                    feature_names=df.columns.values,
                    class_names=['0', '1'], rounded=True,
                    proportion=True, label='root',
                    precision=2, filled=True)
    with open("tree.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)
    # http://www.webgraphviz.com/
    return

def decisionTreeTuning():
    # Parameter estimation using grid search with cross-validation
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
    X_train, y_train, X_val, y_val, test_train, test_val = getAllCleanedData(binning=1)
    X_train_dummies = pd.get_dummies(X_train)
    X_val_dummies = pd.get_dummies(X_val)
    test_train_dummies = pd.get_dummies(test_train)

    printFullDf(X_train_dummies.head())

    params = {
        "criterion": ("gini", "entropy"),
        "splitter": ("best", "random"),
        "max_depth": (list(range(3, 10))),
        "min_samples_split": list(range(2, 15)),
        "min_samples_leaf": list(range(1, 10)),
    }
    # To obtain a deterministic behaviour during fitting, we set random_state to a fixed int

    model = DecisionTreeClassifier(random_state=0)
    grid_search_cv = GridSearchCV(model, params, scoring="f1", verbose=1, n_jobs=-1, cv=3)
    grid_search_cv.fit(X_train_dummies, y_train)
    best_grid = grid_search_cv.best_estimator_
    print(best_grid)
    print_score(grid_search_cv, X_train_dummies, y_train, X_val, y_val, train=True)
    print_score(grid_search_cv, X_train_dummies, y_train, X_val_dummies, y_val, train=False)
    print_score(grid_search_cv, X_train_dummies, y_train, test_train_dummies, test_val, train=False)
    # visualizeTree(tree, X)
    # print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    # print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    # ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    # ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    # visualizeTree(grid_search_cv, X_train_dummies)
    return

decisionTree()