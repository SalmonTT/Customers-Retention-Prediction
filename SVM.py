from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from dataPreprocessing import *
from utils import *

def runSVM(X_train, X_test, y_train, y_test):
    svm = SVC(kernel='rbf', class_weight='balanced', gamma='auto', probability=True)
    svm.fit(X_train, y_train)
    print_score(svm, X_train, y_train, X_test, y_test, train=True)
    print_score(svm, X_train, y_train, X_test, y_test, train=False)
    ROC(svm, X_train, y_train, X_test, y_test, train=True)
    ROC(svm, X_train, y_train, X_test, y_test, train=False)

def tuneSVM(X_train, X_test, y_train, y_test):
    C = [0.001, 0.01, 0.1, 0.2, 0.5, 0.8]
    SVMtrainAcc = []
    SVMtestAcc = []

    # a for loop to see the results varying parameter C.
    for param in C:
        # clf = SVC(C=param,kernel='linear',gamma='auto') # Linear Support Vector Machine
        clf = SVC(C=param, kernel='rbf',
                  gamma='auto')  # Nonlinear Support Vector Machine rbf is more powerful than linear
        clf.fit(X_train, y_train)
        Y_predTrain = clf.predict(X_train)
        Y_predTest = clf.predict(X_test)
        SVMtrainAcc.append(accuracy_score(y_train, Y_predTrain))
        SVMtestAcc.append(accuracy_score(y_test, Y_predTest))

    plt.plot(C, SVMtrainAcc, 'ro-', C, SVMtestAcc, 'bv--')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('C')
    plt.xscale('log')
    plt.ylabel('Accuracy')
    plt.show()

def useSVM():
    train = getTrainingData('train.csv', visualize=False)
    X = train.drop(['Exited'], axis=1)
    standard(X)
    y = train.Exited
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    runSVM(X_train, X_test, y_train, y_test)
    # tuneSVM(X_train, X_test, y_train, y_test)

useSVM()