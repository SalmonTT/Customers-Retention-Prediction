from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import graphviz
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib as plt
from utils import print_score, ROC

def decisionTreeTuned(X_train, X_test, y_train, y_test, df):
    # Tuning the hyper-parameters
    params = {
        "criterion": ("gini", "entropy"),
        "splitter": ("best", "random"),
        "max_depth": (list(range(1, 10))),
        "min_samples_split": list(range(10, 30)),
        "min_samples_leaf": list(range(1, 20)),
    }

    model = DecisionTreeClassifier(random_state=42)
    # n_jobs = -1: using all processors, cv=5: 5 cross-validation
    grid_search_cv = GridSearchCV(model, params, scoring="accuracy", verbose=1, n_jobs=-1, cv=3)
    grid_search_cv.fit(X_train, y_train)
    print(grid_search_cv.best_params_)
    # {'criterion': 'entropy', 'max_depth': 17, 'min_samples_leaf': 2, 'min_samples_split': 28, 'splitter': 'random'}

    # grid.search_cv.best_estimator_ is a DecisionTreeClassifier(max_depth=6, random_state=42, splitter='random')
    print_score(grid_search_cv.best_estimator_, X_train, y_train, X_test, y_test, train=True)
    print_score(grid_search_cv.best_estimator_, X_train, y_train, X_test, y_test, train=False)
    ROC(grid_search_cv.best_estimator_, X_train, y_train, X_test, y_test, train=True)
    ROC(grid_search_cv.best_estimator_, X_train, y_train, X_test, y_test, train=False)

    # visualizing tree
    export_graphviz(grid_search_cv, out_file='tree.dot',
                    feature_names=df.columns.values,
                    class_names=['0', '1'], rounded=True,
                    proportion=True, label='root',
                    precision=2, filled=True)
    with open("tree.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)
    # http://www.webgraphviz.com/

    return grid_search_cv

def decisionTree(X_train, X_test, y_train, y_test, df):
    tree = DecisionTreeClassifier(criterion='gini',
                                  max_depth=7, max_features=None, max_leaf_nodes=None,
                                  min_impurity_decrease=0.0, min_impurity_split=None,
                                  min_samples_leaf=1, min_samples_split=2,
                                  random_state=42, splitter='random')
    tree.fit(X_train, y_train)

    print_score(tree, X_train, y_train, X_test, y_test, train=True)
    print_score(tree, X_train, y_train, X_test, y_test, train=False)
    ROC(tree, X_train, y_train, X_test, y_test, train=True)
    ROC(tree, X_train, y_train, X_test, y_test, train=False)

    # visualizing tree
    export_graphviz(tree, out_file='tree.dot',
                    feature_names=df.columns.values,
                    class_names=['0', '1'], rounded=True,
                    proportion=True, label='root',
                    precision=2, filled=True)
    with open("tree.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)
    # http://www.webgraphviz.com/
    return tree

def useDecisionTree(df):
    # ----- split training data -----
    X = df.drop(['Exited'], axis=1)
    y = df.Exited
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = decisionTree(X_train, X_test, y_train, y_test, X)
    return model
