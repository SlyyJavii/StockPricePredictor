import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

def lstm_train_predict(x_train, y_train, x_test):

    #scale the data
    scaler_X = MinMaxScaler(feature_range=(-1,1))
    scaler_Y = MinMaxScaler(feature_range=(-1,1))

    X_train_scaled = scaler_X.fit_transform(x_train)
    X_test_scaled = scaler_X.transform(x_test)

    Y_train_scaled = scaler_Y.fit_transform(y_train.values.reshape(-1,1))

    #reshape the data to make it from 2d to 3d
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    #create the LSTM model
    model = Sequential([
        tf.keras.layers.Input(shape = (X_train_scaled.shape[1], 1)),
        #first LSTM layer
        LSTM(200, activation = 'relu', return_sequences = True),
        Dropout(0.2),
        #secons LSTM layer
        LSTM(100, activation = 'relu', return_sequences = True), 
        Dropout(0.2),
        #third LSTM layer
        LSTM(100, activation = 'relu'),
        Dropout(0.2),
        #output layer
        Dense(1)
    ])

    model.compile(optimizer = 'adam', loss = 'mse')

    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)

    #fit/train the model
    model.fit(X_train_scaled, Y_train_scaled, epochs = 250, batch_size = 128, validation_split = 0.2,callbacks = [early_stopping])

    #predict the data
    predictions_scaled = model.predict(X_test_scaled)

    #inverse the scaling to get actual predictions
    predictions = scaler_Y.inverse_transform(predictions_scaled)

    return predictions