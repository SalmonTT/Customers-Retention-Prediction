import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python import keras
from tensorflow.keras.layers import Input, Dense, SimpleRNN, RNN
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence

from dataPreprocessing import getTrainingData


def rnn(X_train, X_test, y_train, y_test):
    batchs_size = 32
    inputs = Input(batch_shape=(batchs_size, 28, 28))
    RNN1 = SimpleRNN(units=128, activation='tanh', return_sequences=False, return_state=False)
    RNN1_output = RNN1(inputs)
    Dense1 = Dense(units=128, activation='relu')(RNN1_output)
    Dense2 = Dense(units=64, activation='relu')(Dense1)
    output = Dense(units=10, activation='softmax')(Dense2)
    rnn = Model(inputs=inputs, outputs=output)
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    rnn.compile(optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    rnn.summary()
    rnn.fit(X_train, y_train, epochs=1, batch_size=batchs_size, validation_data=(X_test, y_test))
    rnn.summary()

    def useRnn():
        train = getTrainingData('train.csv', visualize=False)
        X = train.drop(['Exited'], axis=1)
        sc = StandardScaler()
        X = sc.fit_transform(X)
        y = train.Exited
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        rnn(X_train, X_test, y_train, y_test)

    useRnn()