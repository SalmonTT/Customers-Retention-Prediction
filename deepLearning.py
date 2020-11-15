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
        metrics="BinaryAccuracy"
    )
    return classifier

def useKeras(df):
    # ----- split training data -----
    X = df.drop(['Exited'], axis=1)
    y = df.Exited
    # ----- standardizing the input features -----
    sc = StandardScaler()
    X = sc.fit_transform(X)
    # ----- random_state is a seed for random sampling -----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # ---- PCA ------
    pca = PCA(n_components='mle')
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    # ----- building the model -----
    model = kerasModel()
    # Fitting the data to the training dataset
    model.fit(
        X_train,
        y_train,
        batch_size=10,
        epochs=60,
    )
    # loss and accuracy
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