from decisionTree import decisionTree
from deepLearning import *
import XGModel as xg
from dataPreprocessing import getTrainingData

from sklearn import model_selection
from sklearn.metrics import classification_report
import pandas as pd

def main():
    # # ----- Part 1: Data Pre-processing ----- #
    # train = getTrainingData('train.csv', visualize=True)
    # # ----- Part 2: data splitting ----- #
    # if 'Exited' in train.columns: # task 2
    #     X = train.drop(['Exited'], axis=1)
    #     # X = StandardScaler().fit_transform(X)
    #     y = train.Exited
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # else:
    #     print("splitting for task 1")
    #
    # # ----- Part 3: comparing all models ----- #
    # models = [
    #     ('tree', decisionTree())
    # ]
    # scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    # # ----- Part 4: output test result ----- #
    # results = []
    # names = []
    # dfs = []
    # for name, model in models:
    #     kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
    #     cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
    #     clf = model.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)
    #     print(name)
    #     print(classification_report(y_test, y_pred, target_names=[0,1]))
    #     results.append(cv_results)
    #     names.append(name)
    #     this_df = pd.DataFrame(cv_results)
    #     this_df['model'] = name
    #     dfs.append(this_df)
    #
    # final_results = pd.concat(dfs, ignore_index=True)
    # print(final_results)
    xg.xgModel()
    return

if __name__ == "__main__":
    main()
