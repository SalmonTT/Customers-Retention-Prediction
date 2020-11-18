from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from dataPreprocessing import *
from utils import *
from imblearn.over_sampling import SMOTE

def randomForest(X_train, X_test, y_train, y_test):
    # Parameter tuning
    model = RandomForestClassifier()
    params = {'max_depth': [5,10,15],
             'min_samples_leaf': [2,4,6,8],
             'n_estimators': [80,120, 140, 200]}

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
    train = getTrainingData('train.csv', visualize=False, discrete=True, encoding=True)
    X_train = train.drop(['Exited'], axis=1)
    y_train = train.Exited
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    df_test = getTestingData(True, True)
    X_test = df_test.drop(['Exited'], axis=1)
    y_test = df_test.Exited

    randomForest(X_train, X_test, y_train, y_test)
    # TuneRandomForest(X_train, X_test, y_train, y_test)

useRandomForest()