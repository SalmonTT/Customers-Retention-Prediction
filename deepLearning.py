from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from dataPreprocessing import *
def kerasModel(X_train, X_test, y_train, y_test, df):
    # https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1
    classifier = Sequential()
    # First Hidden Layer
    classifier.add(Dense(6, activation='relu', kernel_initializer='random_normal', input_dim=14))
    # Second  Hidden Layer
    classifier.add(Dense(6, activation='relu', kernel_initializer='random_normal'))
    # Output Layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    # Compiling the neural network
    classifier.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['BinaryAccuracy'])
    # Fitting the data to the training dataset
    classifier.fit(X_train, y_train, batch_size=10, epochs=100)

    # loss and accuracy
    eval_model = classifier.evaluate(X_train, y_train)
    # print(eval_model)
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("Accuracy score: ", accuracy_score(y_test, y_pred))
    print("Precision score: ", precision_score(y_test, y_pred))
    print("Recall score: ", recall_score(y_test, y_pred))
    print("F1 score: ", f1_score(y_test, y_pred))

    return

def useKeras(df):
    # ----- split training data -----
    X = df.drop(['Exited'], axis=1)
    y = df.Exited
    # ----- standardizing the input features -----
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # random_state is a seed for random sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    kerasModel(X_train, X_test, y_train, y_test, X)
    return

def tune():
    df = getTrainingData('Train.csv', False)
    useKeras(df)

tune()