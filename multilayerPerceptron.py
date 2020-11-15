from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from dataPreprocessing import *
from utils import *

def multilayerPerceptron(X_train, X_test, y_train, y_test):
    # params = {
    #     "hidden_layer_sizes": [(100,),(100,50),(50,25),(50,50),(100,100),(100,50,25)],
    #     "activation": ['logistic'],
    #     "solver": ['adam'],
    #     "verbose": [True]
    # }
    # model = MLPClassifier()
    # grid_search_cv = GridSearchCV(model, params, verbose=1, n_jobs=-1, cv=3, scoring='roc_auc')
    # grid_search_cv.fit(X_train, y_train)
    # best_grid = grid_search_cv.best_estimator_
    # print(best_grid)
    # print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    # print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    # ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    # ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=False)

    mlp = MLPClassifier(activation = 'logistic', solver='adam', alpha=1e-5, hidden_layer_sizes=(100,100))
    # mlp = MLPClassifier(activation='relu', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50))
    mlp.fit(X_train, y_train)
    print_score(mlp, X_train, y_train, X_test, y_test, train=True)
    print_score(mlp, X_train, y_train, X_test, y_test, train=False)
    ROC(mlp, X_train, y_train, X_test, y_test, train=True)
    ROC(mlp, X_train, y_train, X_test, y_test, train=False)


def useMultilayerPerceptron():
    train = getTrainingData('train.csv', visualize=False)
    X = train.drop(['Exited'], axis=1)
    y = train.Exited
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_test = standard(X_train, X_test)
    multilayerPerceptron(X_train, X_test, y_train, y_test)


useMultilayerPerceptron()