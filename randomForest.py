from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from dataPreprocessing import *
from utils import *

def randomForest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100, criterion = 'gini', max_samples=0.5,
                               max_depth = 10, random_state=42)
    rf.fit(X_train, y_train)
    print_score(rf, X_train, y_train, X_test, y_test, train=True)
    print_score(rf, X_train, y_train, X_test, y_test, train=False)
    ROC(rf, X_train, y_train, X_test, y_test, train=True)
    ROC(rf, X_train, y_train, X_test, y_test, train=False)


def useRandomForest():
    train = getTrainingData('train.csv', visualize=False)
    X = train.drop(['Exited'], axis=1)
    standard(X)
    y = train.Exited
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    randomForest(X_train, X_test, y_train, y_test)

useRandomForest()