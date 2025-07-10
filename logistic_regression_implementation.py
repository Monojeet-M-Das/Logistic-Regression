import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
import pandas as pd
import yfinance as yf

def download_data(stock, start, end):
    data = {}
    ticker = yf.download(stock, start, end)
    data['close'] = ticker['Close'].squeeze()
    return pd.DataFrame(data)

def construct_features(data, lags=2):

    # calculate the lagged closing prices (name=close)
    for i in range(0, lags):
        data['Lag%s'%str(i+1)] = data['close'].shift(i+1)
        
    # calculate the percent actual change
    data['Today Change'] = data['close']
    data['Today Change'] = data['Today Change'].pct_change() * 100

    # calculate the lags in percentage (normalization)
    for i in range(0, lags):
        data['Lag%s'%str(i+1)] = data['Lag%s'%str(i+1)].pct_change() * 100

    # direction - the target variable
    data['Direction'] = np.where(data['Today Change'] > 0, 1, -1)

if __name__ == '__main__':
    start = datetime.datetime(2017,1,1)
    end = datetime.datetime(2018,1,1)
    
    stock_data = download_data('IBM', start, end)
    construct_features(stock_data)
    stock_data.dropna(inplace=True)

    # features and labels (target variables)
    X = stock_data[['Lag1', 'Lag2']]
    y = stock_data['Direction']

    # split the datat into tarining and test set (70% / 30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # training the model on the training set
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test the model on the test set
    predictions = model.predict(X_test)

    print('Accurace of the model: %.2f' % accuracy_score(y_test, predictions))
    print(confusion_matrix(predictions, y_test))