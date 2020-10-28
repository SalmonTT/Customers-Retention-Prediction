from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split

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

def kerasModel(X_train, X_test, y_train, y_test, df):
    # https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1
    return