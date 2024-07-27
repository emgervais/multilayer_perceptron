from sklearn.preprocessing import OneHotEncoder
from MLP import MLP
import pandas as pd
import joblib

def extractData():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    X_train = train_df.drop(columns=['y']).values
    y_train = train_df['y'].values
    X_test = test_df.drop(columns=['y']).values
    y_test = test_df['y'].values

    return X_train, y_train, X_test, y_test

def main():
    
    X_train, y_train, X_test, y_test = extractData()
    # onehot_encoder = OneHotEncoder(sparse_output=False)
    # y_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
    # y_test = onehot_encoder.fit_transform(y_test.reshape(-1, 1))

    model = MLP(layers=[X_train.shape[1], 16, 16, 1], activation='relu')
    model.fit(X_train, y_train, X_test, y_test, epochs=1000, learning_rate=0.0001)
    joblib.dump(model, 'mlp_model.pkl')

if __name__ == '__main__':
    main()