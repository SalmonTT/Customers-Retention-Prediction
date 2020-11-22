from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import graphviz
from sklearn.model_selection import GridSearchCV, train_test_split
from utils import print_score, ROC
from dataPreprocessing import *

def decisionTree():
    # this always uses the best parameters
    tree = DecisionTreeClassifier(criterion='gini',
                                  max_depth=7, max_features=None, max_leaf_nodes=None,
                                  min_impurity_decrease=0.0, min_impurity_split=None,
                                  min_samples_leaf=1, min_samples_split=2,
                                  random_state=42, splitter='best')
    return tree

def visualizeTree(tree, df):
    export_graphviz(tree, out_file='tree.dot',
                    feature_names=df.columns.values,
                    class_names=['0', '1'], rounded=True,
                    proportion=True, label='root',
                    precision=2, filled=True)
    with open("tree.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)
    # http://www.webgraphviz.com/
    return

def decisionTreeTuning():
    # Parameter estimation using grid search with cross-validation
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
    X_train, y_train, X_val, y_val, test_train, test_val = getAllCleanedData(standardize=0, binning=1)
    params = {
        "criterion": ("gini", "entropy"),
        "splitter": ("best", "random"),
        "max_depth": (list(range(1, 10))),
        "min_samples_split": list(range(2, 15)),
        "min_samples_leaf": list(range(1, 10)),
    }
    # To obtain a deterministic behaviour during fitting, we set random_state to a fixed int

    model = DecisionTreeClassifier(random_state=0)
    grid_search_cv = GridSearchCV(model, params, scoring="accuracy", verbose=1, n_jobs=-1, cv=3)
    grid_search_cv.fit(X_train, y_train)
    print_score(grid_search_cv, X_train, y_train, X_val, y_val, train=True)
    print_score(grid_search_cv, X_train, y_train, X_val, y_val, train=False)
    print_score(grid_search_cv, X_train, y_train, test_train, test_val, train=False)
    # visualizeTree(tree, X)
    # print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    # print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    # ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=True)
    # ROC(grid_search_cv, X_train, y_train, X_test, y_test, train=False)
    # visualizeTree(grid_search_cv, X)
    return

decisionTreeTuning()