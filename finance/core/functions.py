import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def preprocess_data(data, n=3, target_column='Close'):
    """
    Convert time series data into a supervised learning format suitable for LSTM training.

    Parameters:
        data (pd.DataFrame): Time series data.
        n (int): Number of lag time steps to include as features.
        target_column (str): Name of the column to predict, default is 'Close'.

    Returns:
        X (np.array): The input features for the model.
        y (np.array): The target variable for the model.
    """

    n_features = data.shape[1]  # Number of features

    agg_df = pd.DataFrame()

    for i in range(n, -1, -1):
        shift_df = data.shift(i)
        shift_df.columns = [f'{col} (t)' if i == 0 else f'{col} (t-{i})' for col in data.columns]
        agg_df = pd.concat([agg_df, shift_df], axis=1)

    # Drop rows with NaN values
    agg_df.dropna(inplace=True)

    # Split into input and outputs
    X, y = agg_df.iloc[:, :n * n_features], agg_df[target_column + ' (t)']

    # In an LSTM the input for each layer needs to contain:
    # - Number of observations
    # - The time steps
    # - The features

    X = X.values.reshape((-1, n, n_features))

    return X, y


def lstm_model(X_train, y_train, X_test, y_test):
    """
    Train an LSTM model on the provided data.

    Parameters:
        X_train (np.array): Input features.
            - X.shape = (Number of observations, time steps, features)
        y_train (np.array): Target variable.
        X_test (np.array): Validation set Input variables
        y_test (np.array): Validation set Target variable.

    Returns:
        model (keras.Model): Trained Keras model.
    """
    # Define LSTM model
    model = Sequential()
    model.add(LSTM(5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit model
    model.fit(X_train, y_train, epochs=30, batch_size=24, verbose=2, validation_data=(X_test, y_test))

    return model
