from finance import __version__
import pytest
import pandas as pd
import numpy as np

from finance.core.functions import preprocess_data


def test_version():
    assert __version__ == '0.1.0'


# Fixture to create a sample DataFrame
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'Close': [27.4, 27.68, 28.11, 28.5, 29.0],
        'simple_rtn': [0.023534, 0.010219, 0.015535, 0.013891, 0.017540],
        'log_rtn': [0.023261, 0.010167, 0.015415, 0.013781, 0.017389]
    }, index=pd.to_datetime(['2012-01-04', '2012-01-05', '2012-01-06', '2012-01-07', '2012-01-08']))
    return data


def test_preprocess_data(sample_data):
    # Run preprocessing
    n = 3  # Number of time steps
    X, y = preprocess_data(sample_data, n, 'Close')

    # Assertions to ensure correct shape and type of output
    assert X.shape == (2, n,
                       3), ("X shape should be (2, n, 3) where 2 is the number of samples, n is the number of "
                            "time steps, and 3 is the number of features")
    assert y.shape == (2,), "y should have one output per sample"
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"
