from tensorflow.keras.layers import Dense    #for Dense layers
from tensorflow.keras.models import Sequential #for sequential implementation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from dataPreprocessing import *
from dataPreprocessing import getTrainingData
from dataPreprocessing import *
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
import numpy as np
import tensorflow.keras.backend as K
from utils import exportCSV
from imblearn.over_sampling import SMOTE

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def kerasModel():

    # https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1
    classifier = Sequential()
    # First Hidden Layer
    classifier.add(Dense(6, activation='relu', kernel_initializer='random_normal'))
    # Second  Hidden Layer
    classifier.add(Dense(6, activation='relu', kernel_initializer='random_normal'))
    # Output Layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    # Compiling the neural network
    classifier.compile(
        optimizer='sgd',
        loss="binary_crossentropy",
        metrics=[get_f1]
    )
    return classifier

def useKeras(df):
    # ----- split training data -----
    X = df.drop(['Exited'], axis=1)
    y = df.Exited
    # ---- SMOTE ------
    # method 1: technically only for continuous data
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    # method 2: SMOTE-NC (Nominal & Continuous), works for cat and continuous (mixed)
    
    # ----- random_state is a seed for random sampling -----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # ----- standardizing the input features -----
    X_train, X_test = standard(X_train, X_test)


    # ----- building the model -----
    model = kerasModel()

    class_weights = compute_class_weight('balanced', np.unique(y_train),y_train)
    class_weights = {i : class_weights[i] for i in range(2)}
    # Fitting the data to the training dataset
    model.fit(
        X_train,
        y_train,
        batch_size=3,
        epochs=120,
        class_weight=class_weights
    )

    # ----- evaluate test sample and return a .csv ----
    test_data = getTestData('assignment-test.csv', False, False, True)
    sc = StandardScaler().fit(test_data)
    test_data = sc.transform(test_data)
    pred_prob = model.predict(test_data)
    print(pred_prob)
    exportCSV('assignment-test.csv', pred_prob)
    # ----- loss and accuracy -----
    eval_model = model.evaluate(X_train, y_train)
    print(eval_model)
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("Accuracy score: ", accuracy_score(y_test, y_pred))
    print("Precision score: ", precision_score(y_test, y_pred))
    print("Recall score: ", recall_score(y_test, y_pred))
    print("F1 score: ", f1_score(y_test, y_pred))


    return

def tune():
    df = getTrainingData('Train.csv', False, False, True)
    useKeras(df)

tune()