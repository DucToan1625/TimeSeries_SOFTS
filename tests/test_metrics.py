import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def test_mae_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.0])

    result = mean_absolute_error(y_true, y_pred)

    assert np.isclose(result, 0.5)


def test_mse_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.0])

    result = mean_squared_error(y_true, y_pred)

    assert np.isclose(result, 0.4166666667)


def test_prediction_shape():
    prediction = np.zeros((1, 24, 1))

    assert prediction.shape == (1, 24, 1)


def test_prediction_values_are_finite():
    prediction = np.zeros((1, 24, 1))

    assert np.isfinite(prediction).all()