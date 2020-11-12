from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from dataPreprocessing import *
from utils import *

def multilayerNeuralNetwork(X_train, X_test, y_train, y_test):
    mnn = MLPClassifier(hidden_layer_sizes=(100,), solver='adam', activation='logistic', max_iter=200)
    mnn.fit(X_train, y_train)
    print_score(mnn, X_train, y_train, X_test, y_test, train=True)
    print_score(mnn, X_train, y_train, X_test, y_test, train=False)
    ROC(mnn, X_train, y_train, X_test, y_test, train=True)
    ROC(mnn, X_train, y_train, X_test, y_test, train=False)

def useMultilayerNeuralNetwork():
    train = getTrainingData('train.csv', visualize=False)
    X = train.drop(['Exited'], axis=1)
    standard(X)
    y = train.Exited
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    multilayerNeuralNetwork(X_train, X_test, y_train, y_test)

useMultilayerNeuralNetwork()