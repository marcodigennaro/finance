import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def lstm_model(data, n=3):
    """
    Train an LSTM model on the provided data.

    Parameters:
        data (pd.DataFrame): Input data with features.
        n (int): Number of timesteps for the LSTM memory.

    Returns:
        model (keras.Model): Trained Keras model.
    """
    # Assuming the last column is the target variable
    values = data.values
    # Normalize features
    values = values.astype('float32')

    # Convert series to supervised learning format
    df = pd.DataFrame(values)
    columns = [df.shift(i) for i in range(1, n + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.dropna(inplace=True)

    # Split into input and outputs
    n_obs = n * df.shape[1]
    X, y = df.iloc[:, :n_obs], df.iloc[:, -1]
    X = X.values.reshape((X.shape[0], n, df.shape[1]))

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit model
    model.fit(X, y, epochs=50, batch_size=72, verbose=2, shuffle=False)

    return model
