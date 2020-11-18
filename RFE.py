from dataPreprocessing import *
from utils import *
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def logisticReg():
    train = getTrainingData('train.csv', visualize=False, discrete=False, encoding=True)
    X_train = train.drop(['Exited'], axis=1)
    print(X_train.columns.values)
    y_train = train.Exited
    #
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    scale = StandardScaler().fit(X_train)
    X_train = scale.transform(X_train)
    # ---- RFE -----
    rfes = []
    scores = []

    for n in range(1,13):
        rfe = RFE(LogisticRegression(random_state=42), n, 1).fit(X_train, y_train)
        rfes.append(rfe)
        yHat = rfe.predict(X_train)
        scores.append(f1_score(y_train, yHat))
    print(scores)
    print(rfes[4].support_)

    # ['CreditScore' 'Age' 'Tenure' 'Balance' 'NumOfProducts' 'HasCrCard'
    #  'EstimatedSalary' 'France' 'Germany' 'Female' 'Active']
    # [False  True False  True False False False False  True  True  True]

    return

logisticReg()