import pandas as pd
import joblib
import numpy as np

def extract():
    data = pd.read_csv('test.csv')
    X_test = data.drop(columns=['y']).values
    y_test = data['y'].values
    return X_test, y_test

def main():
    X_test, y_test = extract()
    model = joblib.load('mlp_model.pkl')
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f'Validation Accuracy: {accuracy:.2f}')
    model.visualize()

if __name__ == '__main__':
    main()