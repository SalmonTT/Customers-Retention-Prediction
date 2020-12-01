from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from dataPreprocessing import getTrainingData, standard
from utils import *

def gnnb(X_train,X_test, Y_train, Y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    print_score(gnb, X_train, Y_train, X_test, Y_test, train=True)
    print_score(gnb, X_train, Y_train, X_test, Y_test, train=False)

def tuneGnnb(X_train, X_test, y_train, y_test):
    n_estimators = [80,100,120,140,160,180,200]
    GnbtrainAcc = []
    GnbtestAcc = []
    for param in n_estimators:
        gnb = GaussianNB(max_depth=25, min_samples_leaf=8, n_estimators=param)
        gnb.fit(X_train, y_train)
        y_predTrain = gnb.predict(X_train)
        y_predTest = gnb.predict(X_test)
        GnbtrainAcc.append(accuracy_score(y_train, y_predTrain))
        GnbtestAcc.append(accuracy_score(y_test, y_predTest))

    plt.plot(n_estimators, GnbtrainAcc, 'ro-', n_estimators, GnbtestAcc, 'bv--')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('max_depth')
    plt.xscale('log')
    plt.ylabel('Accuracy')
    plt.show()

def useGnnb():
    train = getTrainingData('train.csv', visualize=False)
    X = train.drop(['Exited'], axis=1)
    y = train.Exited
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_test = standard(X_train, X_test)
    gnnb(X_train, X_test, y_train, y_test)

useGnnb()


