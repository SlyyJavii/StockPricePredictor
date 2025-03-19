import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os 

#import the trained models for predictions
from src.lstm import lstm_train_predict
from src.randomForest import randomForest_train_predict
from src.svr import svr_train_predict

#importing data
data = pd.read_csv('StockPricePredictor\data\stocks_data.csv') 

df = pd.DataFrame(data)

#evaluation function
def evaluate_model(model, stock, Y_test, Y_predictions):
    print(f"{model} evaluation for {stock}:")

    mse = mean_squared_error(Y_test, Y_predictions)
    r2 = r2_score(Y_test, Y_predictions)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return r2, mse

#predicting stock
stocks = data['Name'].unique() #getting all different stocks in data
features = ['open', 'high', 'low', 'close', 'volume', 'return', 'rolling_mean', 'rolling_std'] #features to be used for prediction
target = 'close' #target variable

#dictionary to store the results 
results = {'lstm_r2':[], 'lstm_mse':[],
           'randomForest_r2' : [], 'randomForest_mse' : [],
            'svr_r2' : [], 'svr_mse': []}

print("stocks to process:", stocks)
for stock in stocks: #iterating over each stock to be predicted by each model

    print(f"Processing stock: {stock}")
    stock_data = data[data['Name'] == stock] #getting the data for the stock

    X = stock_data[features] #features
    y = stock_data[target] #target
    dates = stock_data['date'] #getting the dates for each stock

    X_train, X_test, Y_train, Y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size = 0.2, random_state = 42)

    #sort test data by date
    X_test_sorted = X_test.sort_index()
    y_test_sorted = Y_test.sort_index()
    dates_test_sorted = pd.to_datetime(dates_test.sort_index())

    #predicting using lstm model
    Y_predictions = lstm_train_predict(X_train, Y_train, X_test_sorted)

    r2, mse = evaluate_model('LSTM', stock, y_test_sorted, Y_predictions)

    results['lstm_r2'].append(r2)
    results['lstm_mse'].append(mse)

    plt.figure(figsize = (10,6))
    plt.plot(dates_test_sorted, y_test_sorted.values, label = 'True Values', color = 'green')
    plt.plot(dates_test_sorted, Y_predictions, label = 'LSTM Prediction', color = 'purple')
    plt.subplots_adjust(bottom=0.2)
    plt.title(f'{stock} - LSTM | R² Score: {r2:.4f} | MSE: {mse:.4f}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.savefig(os.path.join('StockPricePredictor/result/lstm_plots', f'{stock}_lstm.png'))
    plt.close()

    #predicting using random forest model
    Y_predictions = randomForest_train_predict(X_train, Y_train, X_test_sorted)

    r2, mse = evaluate_model('Random Forest', stock, y_test_sorted, Y_predictions)

    results['randomForest_r2'].append(r2)
    results['randomForest_mse'].append(mse)

    plt.figure(figsize = (10,6))
    plt.plot(dates_test_sorted, y_test_sorted.values, label = 'True Values', color = 'green')
    plt.plot(dates_test_sorted, Y_predictions, label = 'Random Forest Predictions', color = 'purple')
    plt.title(f'{stock} - Random Forest | R² Score: {r2:.4f} | MSE: {mse:.4f}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.savefig(os.path.join('StockPricePredictor/result/randomForest_plots', f'{stock}_randomForest.png'))
    plt.close()

    #predicting using svr model
    Y_predictions = svr_train_predict(X_train, Y_train, X_test_sorted)

    r2, mse = evaluate_model('SVR', stock, y_test_sorted, Y_predictions)

    results['svr_r2'].append(r2)
    results['svr_mse'].append(mse)

    plt.figure(figsize = (10,6))
    plt.plot(dates_test_sorted, y_test_sorted.values, label = 'True Values', color = 'green')
    plt.plot(dates_test_sorted, Y_predictions, label = 'SVR Predictions', color = 'purple')
    plt.title(f'{stock} - SVR | R² Score: {r2:.4f} | MSE: {mse:.4f}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.savefig(os.path.join('StockPricePredictor/result/svr_plots', f'{stock}_svr.png'))
    plt.close()

#printing results
print("Results:")

#lstm average results
lstm_r2 = np.mean(results['lstm_r2'])
lstm_mse = np.mean(results['lstm_mse'])
print(f"LSTM Average R2 Score: {lstm_r2:.4f}")
print(f"LSTM average Mean Squared Error: {lstm_mse:.4f}")

#random forest average results
randomForest_r2 = np.mean(results['randomForest_r2'])
randomForest_mse = np.mean(results['randomForest_mse'])
print(f"Random Forest Average R2 Score: {randomForest_r2:.4f}")
print(f"Random Forest average Mean Squared Error: {randomForest_mse:.4f}")

#svr average results
svr_r2 = np.mean(results['svr_r2'])
svr_mse = np.mean(results['svr_mse'])
print(f"SVR Average R2 Score: {svr_r2:0.4f}")
print(f"SVR average mean squared error:{svr_mse:.4f}")
