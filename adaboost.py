from sklearn.ensemble import AdaBoostClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataPreprocessing import *
from utils import *
#not tunned yet

def adaboost(X_train, X_test, y_train, y_test):
    ab = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=42)
    ab.fit(X_train, y_train)
    print_score(ab, X_train, y_train, X_test, y_test, train=True)
    print_score(ab, X_train, y_train, X_test, y_test, train=False)
    ROC(ab, X_train, y_train, X_test, y_test, train=True)
    ROC(ab, X_train, y_train, X_test, y_test, train=False)

def useAdaboost():
    train = getTrainingData('train.csv', visualize=False)
    X = train.drop(['Exited'], axis=1)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = train.Exited
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    adaboost(X_train, X_test, y_train, y_test)

useAdaboost()
