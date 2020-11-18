import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import numpy as np


def printFullDf(df):
    # prints the dataframe in full
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)

def printFullRow(df):
    pd.set_option('display.max_columns', None)
    print(df)

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        print("Train Result:\n===========================================")
        print(f"accuracy score: {accuracy_score(y_train, pred):.4f}\n")
        print(
            f"Classification Report: \n \tPrecision: {precision_score(y_train, pred)}\n\tRecall Score: {recall_score(y_train, pred)}\n\tF1 score: {f1_score(y_train, pred)}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, clf.predict(X_train))}\n")

    elif train == False:
        pred = clf.predict(X_test)
        print("Test Result:\n===========================================")
        print(f"accuracy score: {accuracy_score(y_test, pred)}\n")
        print(
            f"Classification Report: \n \tPrecision: {precision_score(y_test, pred)}\n\tRecall Score: {recall_score(y_test, pred)}\n\tF1 score: {f1_score(y_test, pred)}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

def ROC(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        # predict probabilities
        probs = clf.predict_proba(X_train)
        # keep probabilities for the positive outcome only
        probs = probs[:, 1]
        fpr, tpr, threshold = roc_curve(y_train, probs)
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic Training')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    else:
        # predict probabilities
        probs = clf.predict_proba(X_test)
        # keep probabilities for the positive outcome only
        probs = probs[:, 1]
        fpr, tpr, threshold = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic Testing')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    return

def description(df):
    desc = df.describe()
    printFullRow(desc)

def histogram(df):
    df.hist(figsize=(20, 20))
    plt.show()

def corrAnalysis(df):
    # corr matrix
    corrmat = df.corr()
    plt.figure(figsize=(10,10))
    sns.heatmap(corrmat, cmap='viridis', annot=True, linewidths=0.5)
    plt.show()

def exportCSV(filename, pred_prob):
    df = pd.read_csv(filename, header=0)
    result = pd.DataFrame()
    result['RowNumber'] = df['RowNumber']
    result['Exited'] = pred_prob
    result['Exited'] = result['Exited'].apply(lambda x: 1 if x >= 0.5 else 0)
    result.to_csv('submission_2.csv', index=False)
    return

def getAnswer():
    df_ans = pd.DataFrame()
    df_test = pd.read_csv('assignment-test.csv', header=0)
    df_full = pd.read_csv('Churn_Modelling.csv', header=0)
    df_ans['RowNumber'] = df_test['RowNumber']
    list_ID = df_test['CustomerId'].tolist()
    list_class = []
    for id in list_ID:
        df = df_full[df_full['CustomerId'] == id]
        ans = df.iloc[0]['Exited']
        list_class.append(ans)
    df_ans['Exited'] = np.array(list_class)
    df_ans.to_csv('submission_ans.csv', index=False)
    return


getAnswer()