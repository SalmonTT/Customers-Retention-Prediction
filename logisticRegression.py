from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from dataPreprocessing import *
from utils import *

def logisticRegression(X_train, X_test, y_train, y_test):
    lm = LogisticRegression(class_weight = 'auto')
    lm.fit(X_train, y_train)
    print_score(lm, X_train, y_train, X_test, y_test, train=True)
    print_score(lm, X_train, y_train, X_test, y_test, train=False)
    ROC(lm, X_train, y_train, X_test, y_test, train=True)
    ROC(lm, X_train, y_train, X_test, y_test, train=False)


def useLogisticRegression():
    train = getTrainingData('train.csv', visualize=False)
    X = train.drop(['Exited'], axis=1)
    standard(X)
    y = train.Exited
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    logisticRegression(X_train, X_test, y_train, y_test)

useLogisticRegression()