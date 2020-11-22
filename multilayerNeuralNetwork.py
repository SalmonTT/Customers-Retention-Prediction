from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from dataPreprocessing import *
from utils import *
from imblearn.over_sampling import SMOTE

def multilayerNeuralNetwork(X_train, X_test, y_train, y_test):
    mnn = MLPClassifier(hidden_layer_sizes=(100,), solver='adam', activation='logistic', max_iter=200)
    mnn.fit(X_train, y_train)
    print_score(mnn, X_train, y_train, X_test, y_test, train=True)
    print_score(mnn, X_train, y_train, X_test, y_test, train=False)
    ROC(mnn, X_train, y_train, X_test, y_test, train=True)
    ROC(mnn, X_train, y_train, X_test, y_test, train=False)

def useMultilayerNeuralNetwork():
    train = getTrainingData('train.csv', visualize=False)
    X = train.drop(['Exited'], axis=1)
    # standard(X)
    y = train.Exited
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scale = StandardScaler().fit(X_train)
    X_train = scale.transform(X_train)
    X_test = scale.transform(X_test)
    multilayerNeuralNetwork(X_train, X_test, y_train, y_test)

def multilayerNN():
    train = getTrainingData('train.csv', visualize=False, discrete=False, encoding=True)
    X_train = train.drop(['Exited'], axis=1)
    y_train = train.Exited
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    scale = StandardScaler().fit(X_train)
    X_train = scale.transform(X_train)

    model = MLPClassifier()
    params = {'hidden_layer_sizes': [ (100,50)],
              'solver': ['adam'],
              'activation': ['logistic']
              }

    grid_search_cv = GridSearchCV(model, params, verbose=1, n_jobs=-1, cv=3, scoring='f1')
    grid_search_cv.fit(X_train, y_train)
    best_grid = grid_search_cv.best_estimator_
    print(best_grid)
    # ----- evaluate test sample and return a .csv ----

    # ----- get testing data ---- #
    df_test = getTestingData(False, True)
    X_test = df_test.drop(['Exited'], axis=1)
    # ['Age', 'NumOfProducts', 'Active', 'Balance0', 'BalanceMid', 'Germany', 'CreditScore']
    # X_test = X_test[['Age', 'NumOfProducts', 'Active', 'Balance', 'Germany', 'CreditScore']]
    y_test = df_test.Exited
    X_test = scale.transform(X_test)
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    # ----- export csv ------
    # test_data = getTestData('assignment-test.csv', False, True, True)
    # pred_prob = grid_search_cv.predict(test_data)
    # print(pred_prob)
    # exportCSV('assignment-test.csv', pred_prob)

    return


multilayerNN()