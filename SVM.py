from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from dataPreprocessing import *
from utils import *

def runSVM(X_train, X_test, y_train, y_test):
    # params = {
    #     "C": [2,  4, 6],
    #     "kernel": ['rbf'],
    # }
    # model = SVC()
    # grid_search_cv = GridSearchCV(model, params, verbose=1, n_jobs=-1, cv=3, scoring='accuracy')
    # grid_search_cv.fit(X_train, y_train)
    # print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    # print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    # results = pd.DataFrame(grid_search_cv.cv_results_)
    # printFullRow(results[results['rank_test_score'] == 1])
    # return

    # svm = SVC(C=3, kernel='rbf', gamma='auto', probability=True)
    svm = SVC(C=8, kernel='poly', gamma='auto',  probability=True)
    svm.fit(X_train, y_train)
    print_score(svm, X_train, y_train, X_test, y_test, train=True)
    print_score(svm, X_train, y_train, X_test, y_test, train=False)
    # ROC(svm, X_train, y_train, X_test, y_test, train=True)
    # ROC(svm, X_train, y_train, X_test, y_test, train=False)

def tuneSVM(X_train, X_test, y_train, y_test):
    degree = [2,3,4,5]
    SVMtrainAcc = []
    SVMtestAcc = []

    # a for loop to see the results varying parameter C.
    for param in degree:
        # clf = SVC(C=param,kernel='linear',gamma='auto') # Linear Support Vector Machine
        # clf = SVC(C=3, kernel='rbf',gamma='auto', coef0=param)  # Nonlinear Support Vector Machine rbf is more powerful than linear
        clf = SVC(C=8, kernel='poly', gamma='auto', degree=param, probability=True)
        clf.fit(X_train, y_train)
        Y_predTrain = clf.predict(X_train)
        Y_predTest = clf.predict(X_test)
        SVMtrainAcc.append(accuracy_score(y_train, Y_predTrain))
        SVMtestAcc.append(accuracy_score(y_test, Y_predTest))

    plt.plot(degree, SVMtrainAcc, 'ro-', degree, SVMtestAcc, 'bv--')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('degree')
    plt.xscale('log')
    plt.ylabel('Accuracy')
    plt.show()

def useSVM():
    train = getTrainingData('train.csv', visualize=False)
    X = train.drop(['Exited'], axis=1)
    y = train.Exited
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_test = standard(X_train, X_test)
    runSVM(X_train, X_test, y_train, y_test)
    # tuneSVM(X_train, X_test, y_train, y_test)

