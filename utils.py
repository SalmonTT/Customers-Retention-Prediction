import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def printFullDf(df):
    # prints the dataframe in full
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)

def printFullRow(df):
    pd.set_option('display.max_columns', None)
    print(df)

def corrTest(df):
    # Part 1 -----------
    # Test the correlation between attributes/features (attributes and features will be used interchangeably):
    # drop the following column from dataframe
    feature = df.drop(['CustomerId', 'Exited'], axis=1)
    feature_corr = feature.corr(method='pearson')
    # print the correlation matrix
    printFullDf(feature_corr)
    # show the correlation heatmap
    plt.matshow(feature_corr)
    plt.show()

    # Part 2 -----------
    # we test the correlation between features and output ('Exited')
    feature_output = df.drop(['CustomerId'], axis=1)
    feature_output_corr = feature_output.corr(method='pearson')
    output_corr = abs(feature_output_corr['Exited'])
    printFullDf(output_corr)

    return

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

def histogram(df):
    df.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20))
    plt.show()
    return