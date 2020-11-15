from sklearn import neighbors
from dataPreprocessing import getTrainingData
from sklearn.model_selection import train_test_split, GridSearchCV
from utils import print_score, ROC, printFullRow
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def knnTuning():
    train = getTrainingData('train.csv', visualize=False)
    X = train.drop(['Exited'], axis=1)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = train.Exited
    # split training data half half
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    params = {
        "n_neighbors": list(range(5,131,2)),
        "weights": ['uniform', 'distance']
    }
    model = neighbors.KNeighborsClassifier()
    grid_search_cv = GridSearchCV(model, params, verbose=1, n_jobs=-1, cv=3,scoring='accuracy')
    # print(grid_search_cv.best_params_)
    grid_search_cv.fit(X_train, y_train)
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    results = pd.DataFrame(grid_search_cv.cv_results_)
    printFullRow(results[results['rank_test_score'] == 1])
    # best param setting: n_neighbors == 11/13, p ==2, weights = distance
    return

def knnRadiusTuning():
    train = getTrainingData('train.csv', visualize=False)
    X = train.drop(['Exited'], axis=1)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = train.Exited
    # split training data half half
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    params = {
        "radius": list(np.arange(5.0, 8.0, 0.1)),
        "weights": ['uniform', 'distance']
    }
    model = neighbors.RadiusNeighborsClassifier()
    grid_search_cv = GridSearchCV(model, params, verbose=1, n_jobs=-1, cv=3, scoring='accuracy')
    # print(grid_search_cv.best_params_)
    grid_search_cv.fit(X_train, y_train)
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    results = pd.DataFrame(grid_search_cv.cv_results_)
    printFullRow(results[results['rank_test_score'] == 1])
    # best param setting: n_neighbors == 11/13, p ==2, weights = distance
    return
# knnRadiusTuning()
# knnTuning()