from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from dataPreprocessing import *
from utils import *
from imblearn.over_sampling import SMOTE

def randomForest(X_train, X_test, y_train, y_test):
    # Parameter tuning
    model = RandomForestClassifier()
    params = {'max_depth': [5,6,7],
             'min_samples_leaf': [2,4,6,8],
             'n_estimators': [120],
              'max_features':['sqrt']}

    grid_search_cv = GridSearchCV(model, params, verbose=1, n_jobs=-1, cv=3, scoring='f1')
    grid_search_cv.fit(X_train, y_train)
    best_grid = grid_search_cv.best_estimator_
    print(best_grid)
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    # ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    # ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=False)


    # tuned with the best parameters
    # rf = RandomForestClassifier(max_depth=10, min_samples_leaf=8, n_estimators=200)
    # rf = RandomForestClassifier(max_depth=10, min_samples_leaf=6, n_estimators=200, oob_score=True)
    # rf.fit(X_train, y_train)
    # print_score(rf, X_train, y_train, X_test, y_test, train=True)
    # print_score(rf, X_train, y_train, X_test, y_test, train=False)
    # ROC(rf, X_train, y_train, X_test, y_test, train=True)
    # ROC(rf, X_train, y_train, X_test, y_test, train=False)

def TuneRandomForest(X_train, X_test, y_train, y_test):
    n_estimators = [150,175,200,225,250]
    RFtrainAcc = []
    RFtestAcc = []
    for param in n_estimators:
        rf = RandomForestClassifier(max_depth=10, min_samples_leaf=6, n_estimators=param)
        rf.fit(X_train, y_train)
        y_predTrain = rf.predict(X_train)
        y_predTest = rf.predict(X_test)
        RFtrainAcc.append(accuracy_score(y_train, y_predTrain))
        RFtestAcc.append(accuracy_score(y_test, y_predTest))

    plt.plot(n_estimators, RFtrainAcc, 'ro-', n_estimators, RFtestAcc, 'bv--')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('max_depth')
    plt.xscale('log')
    plt.ylabel('Accuracy')
    plt.show()


def useRandomForest():

    X_train, y_train, X_val, y_val, test_train, test_val = getAllCleanedData(standardize=0, binning=1)
    randomForest(X_train, X_val, y_train, y_val)
    # TuneRandomForest(X_train, X_test, y_train, y_test)

    train = getTrainingData('train.csv', visualize=False, discrete=False, encoding=True)
    X_train = train.drop(['Exited'], axis=1)
    y_train = train.Exited
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    scale = StandardScaler().fit(X_train)
    X_train = scale.transform(X_train)

    # df_test = getTestingData(True, True)
    # X_test = df_test.drop(['Exited'], axis=1)
    # y_test = df_test.Exited
    model = RandomForestClassifier()
    params = {'max_depth': [13,14,15,16],
              'min_samples_leaf': [2],
              'n_estimators': [140,150,160],
              'class_weight': ["balanced"]
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


useRandomForest()