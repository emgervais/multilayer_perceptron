from logistic_regression import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sys

def StandardScaler(X):
    mean = np.array([np.mean(col) for col in X.T])
    std = np.array([np.mean(col) for col in X.T])
    return (X - mean) / std

def train_test_split(X, Y, test_size=0.2, random_state=0):
    if(random_state != 0):
        p = np.random.default_rng(seed=random_state).permutation(len(X))
    else:
        p = np.random.permutation(len(X))

    X_offset = int(len(X) * test_size)
    y_offset = int(len(Y) * test_size)  
    X_train = X[p][X_offset:]
    X_test = X[p][:X_offset]    
    y_train = Y[p][y_offset:]
    y_test = Y[p][:y_offset]
    return (X_train, X_test, y_train, y_test)

def feature_importance(X, y, test_X, test_y):
    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(test_X)
    accuracy = accuracy_score(test_y, y_pred)
    return np.abs(model.coefficients()), accuracy

def RFE(X, y, test_X, test_y, n):
    accuracy_story = []
    num_features = X.shape[1]
    selected_features = list(range(num_features))

    while(len(selected_features) > n):
        importances, accuracy = feature_importance(X[:, selected_features], y, test_X[:, selected_features], test_y)
        least = np.argmin(importances)
        accuracy_story.append(accuracy)
        del selected_features[least]
    return selected_features, accuracy_story

def main():
    data = pd.read_csv('data.csv', header=None)
    data = data.drop(columns=[0])

    columns = ['label'] + [f'feature_{i}' for i in range(1, len(data.columns))]
    data.columns = columns

    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # M=1 & B=0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = StandardScaler(X_train)
    X_test = StandardScaler(X_test)

    selected, _ = RFE(X_train, y_train, X_test, y_test, 13)
    if(len(sys.argv) > 1 and sys.argv[1] == '-v'):
        _, hist = RFE(X_train, y_train, X_test, y_test, 0)
        plt.plot(hist, '-o', mouseover=True)
        plt.show()
    X_train_selected = X_train[:, selected]
    X_test_selected = X_test[:, selected]

    X_t = pd.DataFrame(X_train_selected)
    X_t['y'] = y_train
    X_v = pd.DataFrame(X_test_selected)
    X_v['y'] = y_test
    X_t.to_csv('train.csv', index=False)
    X_v.to_csv('test.csv', index=False)
if __name__ == '__main__':
    main()