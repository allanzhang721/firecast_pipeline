import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

def load_fire_data_from_excel(path):
    df = pd.read_excel(path, engine="openpyxl", header=0)

    # Remove whitespace from column names
    df.columns = [col.strip() for col in df.columns]

    # Drop rows with any NaNs
    df = df.dropna()

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def log_minmax_scale_fire_data(X_train, X_test, y_train, y_test):
    X_train_log, X_test_log = np.log1p(X_train), np.log1p(X_test)
    y_train_log, y_test_log = np.log1p(y_train), np.log1p(y_test)

    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_log)
    X_test_scaled = scaler_X.transform(X_test_log)
    y_train_scaled = scaler_y.fit_transform(y_train_log.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test_log.values.reshape(-1, 1))

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y


import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

def apply_scaling(X_train, X_test, y_train, y_test, scale_mode):
    scaler_X, scaler_y = None, None

    # === Remove constant columns before scaling === #
    variance = X_train.std()
    non_constant_cols = variance[variance != 0].index
    X_train = X_train[non_constant_cols]
    X_test = X_test[non_constant_cols]

    # === Prevent invalid log1p on values < -1 === #
    if scale_mode in ["log", "log_minmax"]:
        # Optional: Clip values to avoid log of negative numbers
        clip_threshold = -0.999  # log1p becomes NaN for values < -1
        X_train = np.clip(X_train, a_min=clip_threshold, a_max=None)
        X_test = np.clip(X_test, a_min=clip_threshold, a_max=None)
        y_train = np.clip(y_train, a_min=clip_threshold, a_max=None)
        y_test = np.clip(y_test, a_min=clip_threshold, a_max=None)

    # === Apply scaling === #
    if scale_mode == "log_minmax":
        X_train = np.log1p(X_train)
        X_test = np.log1p(X_test)
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)

        scaler_X = MinMaxScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        scaler_y = MinMaxScaler()
        y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    elif scale_mode == "log":
        X_train = np.log1p(X_train)
        X_test = np.log1p(X_test)
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)

    elif scale_mode == "minmax":
        scaler_X = MinMaxScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        scaler_y = MinMaxScaler()
        y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    elif scale_mode == "robust":
        scaler_X = RobustScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        scaler_y = RobustScaler()
        y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    elif scale_mode == "original":
        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
        y_test = y_test.values

    else:
        raise ValueError(f"Invalid scale_mode: {scale_mode}")

    # === Final NaN safety check === #
    assert not np.isnan(X_train).any(), "NaN in X_train after scaling"
    assert not np.isnan(X_test).any(), "NaN in X_test after scaling"
    assert not np.isnan(y_train).any(), "NaN in y_train after scaling"
    assert not np.isnan(y_test).any(), "NaN in y_test after scaling"

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y
