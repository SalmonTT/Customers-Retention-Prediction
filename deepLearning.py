from tensorflow.keras.layers import Dense    #for Dense layers
from tensorflow.keras.models import Sequential #for sequential implementation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
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
    classifier.add(Dense(6, activation='relu', kernel_initializer='glorot_uniform'))
    # Second  Hidden Layer
    classifier.add(Dense(6, activation='relu', kernel_initializer='glorot_uniform'))
    # Output Layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform'))
    # Compiling the neural network
    classifier.compile(
        optimizer='sgd',
        loss="binary_crossentropy",
        metrics=[get_f1]
    )
    return classifier

def useKeras(df):
    # ----- split training data -----
    X_train = df.drop(['Exited'], axis=1)
    y_train = df.Exited
    # ----- standardizing the input features -----
    scale = StandardScaler().fit(X_train)
    X_train = scale.transform(X_train)

    # ---- SMOTE ------
    # method 1: technically only for continuous data
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)


    # ----- building the model -----
    model = kerasModel()
    # ----- class weights -----
    # class_weights = compute_class_weight('balanced', np.unique(y_train),y_train)
    # class_weights = {i : class_weights[i] for i in range(2)}
    # Fitting the data to the training dataset
    model.fit(
        X_train,
        y_train,
        batch_size=3,
        epochs=100,
        # class_weight='zeros'
    )

    # ----- evaluate test sample and return a .csv ----
    # test_data = getTestData('assignment-test.csv', False, False, True)
    # sc = StandardScaler().fit(test_data)
    # test_data = sc.transform(test_data)
    # pred_prob = model.predict(test_data)
    # print(pred_prob)
    # exportCSV('assignment-test.csv', pred_prob)

    # ----- get testing data ---- #
    df_test = getTestingData(False, True)
    X_test = df_test.drop(['Exited'], axis=1)
    X_test = scale.transform(X_test)
    y_test = df_test.Exited

    # # ----- loss and accuracy -----
    # eval_model = model.evaluate(X_train, y_train)
    # print(eval_model)
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