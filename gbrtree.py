from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataPreprocessing import *
from utils import *

#not tuned yet
def gbrtree(X_train, X_test, y_train, y_test):
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=10, random_state = 42)
    gbr.fit(X_train, y_train)
    print_score(gbr, X_train, y_train, X_test, y_test, train=True)
    print_score(gbr, X_train, y_train, X_test, y_test, train=False)
    ROC(gbr, X_train, y_train, X_test, y_test, train=True)
    ROC(gbr, X_train, y_train, X_test, y_test, train=False)

def useGbrtree():
    train = getTrainingData('train.csv', visualize=False)
    X = train.drop(['Exited'], axis=1)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = train.Exited
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    gbrtree(X_train, X_test, y_train, y_test)

useGbrtree()

