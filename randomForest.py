from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from dataPreprocessing import *
from utils import *

def randomForest(X_train, X_test, y_train, y_test):
    # Parameter tuning
    model = RandomForestClassifier()
    params = {'max_depth': [5,10,15],
             'min_samples_leaf': [2,4,6,8],
             'n_estimators': [80,120, 140, 200]}

    grid_search_cv = GridSearchCV(model, params, verbose=1, n_jobs=-1, cv=3, scoring='roc_auc')
    grid_search_cv.fit(X_train, y_train)
    best_grid = grid_search_cv.best_estimator_
    print(best_grid)
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=False)


    # tuned with the best parameters
    # rf = RandomForestClassifier(max_depth=10, min_samples_leaf=4, n_estimators=140, random_state=42)
    # rf.fit(X_train, y_train)
    # print_score(rf, X_train, y_train, X_test, y_test, train=True)
    # print_score(rf, X_train, y_train, X_test, y_test, train=False)
    # ROC(rf, X_train, y_train, X_test, y_test, train=True)
    # ROC(rf, X_train, y_train, X_test, y_test, train=False)

def TuneRandomForest(X_train, X_test, y_train, y_test):
    max_depth = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    RFtrainAcc = []
    RFtestAcc = []
    for param in max_depth:
        rf = RandomForestClassifier(max_depth=param , min_samples_leaf=4, n_estimators=140, random_state=42)
        rf.fit(X_train, y_train)
        y_predTrain = rf.predict(X_train)
        y_predTest = rf.predict(X_test)
        RFtrainAcc.append(accuracy_score(y_train, y_predTrain))
        RFtestAcc.append(accuracy_score(y_test, y_predTest))

    plt.plot(max_depth, RFtrainAcc, 'ro-', max_depth, RFtestAcc, 'bv--')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('max_depth')
    plt.xscale('log')
    plt.ylabel('Accuracy')
    plt.show()


def useRandomForest():
    train = getTrainingData('train.csv', visualize=False)
    X = train.drop(['Exited'], axis=1)
    standard(X)
    y = train.Exited
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    randomForest(X_train, X_test, y_train, y_test)
    # TuneRandomForest(X_train, X_test, y_train, y_test)

useRandomForest()