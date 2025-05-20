# 🔥 firecast_pipeline – Updated Function Reference

---

## `load_fire_data_from_excel`

```python
load_fire_data_from_excel(path)
```

Load a fire dataset from an Excel file and split it into features and target.

**Parameters**:
- `path` (`str`) — Path to the `.xlsx` file.

**Returns**:
- `X` (`pd.DataFrame`) — Input features (all columns except the last).
- `y` (`pd.Series`) — Target variable (last column).

---

## `log_minmax_scale_fire_data`

```python
log_minmax_scale_fire_data(X_train, X_test, y_train, y_test)
```

Apply `log1p` transformation followed by MinMax scaling to fire data.

**Parameters**:
- `X_train`, `X_test` (`pd.DataFrame`) — Feature matrices.
- `y_train`, `y_test` (`pd.Series`) — Target variables.

**Returns**:
- `X_train_scaled`, `X_test_scaled`, `y_train_scaled`, `y_test_scaled` (`np.ndarray`)
- `scaler_X`, `scaler_y` — Fitted `MinMaxScaler` instances.

---

## `train_model_for_fire`

```python
train_model_for_fire(X, y, model_name)
```

Train a regression model for fire data using the specified model type.

**Parameters**:
- `X` (`pd.DataFrame`) — Input features.
- `y` (`pd.Series`) — Target variable.
- `model_name` (`str`) — One of `ols`, `lasso`, `mlp`, `xgboost`.

**Returns**:
- Trained model instance (e.g., `LinearRegression`, `Lasso`, `MLPRegressor`, `XGBRegressor`).

---

## `CNNModel`

```python
class CNNModel(nn.Module)
```

A 1D CNN model for regression tasks on tabular data.

**Constructor Parameters**:
- `num_filters1` (`int`) — Filters in the first conv layer.
- `num_filters2` (`int`) — Filters in the second conv layer.
- `fc1_size` (`int`) — Hidden units in the fully connected layer.

**Input**:
- Tensor of shape `(batch_size, 1, num_features)`

**Returns**:
- Tensor of shape `(batch_size, 1)` — Regression prediction.

---

## `train_optuna_cnn_for_fire`

```python
train_optuna_cnn_for_fire(X_train, y_train, X_test, y_test)
```

Train a CNN model using Optuna for hyperparameter tuning.

**Parameters**:
- `X_train`, `X_test` — Scaled feature arrays (`np.ndarray`)
- `y_train`, `y_test` — Scaled target arrays (`np.ndarray`)

**Returns**:
- `best_model` — Trained PyTorch model.
- `metrics` (`dict`) — Evaluation metrics: R², MAE, MSE.

---

## `predict_fire_risk_from_model`

```python
predict_fire_risk_from_model(model_path, input_path)
```

Predict fire risk values using a saved `.joblib` model.

**Parameters**:
- `model_path` (`str`) — Path to trained model (`.joblib`).
- `input_path` (`str`) — Path to `.xlsx` file with test features.

**Returns**:
- `np.ndarray` — Predictions in original units (after inverse log and scaling).

---

## `plot_fire_risk_surface_matplotlib`

```python
plot_fire_risk_surface_matplotlib(model, X_scaled_df, scaler_X, scaler_y, feat1_name, feat2_name, title, save_path="fire_risk_surface.html")
```

Plot and save a 3D prediction surface using Plotly based on two input features.

**Parameters**:
- `model` — Trained PyTorch CNN model.
- `X_scaled_df` (`pd.DataFrame`) — Scaled + log-transformed features.
- `scaler_X`, `scaler_y` — Scalers used during training.
- `feat1_name`, `feat2_name` (`str`) — Names of the X and Y axis features.
- `title` (`str`) — Plot title.
- `save_path` (`str`) — File path to save the interactive `.html` surface plot.

**Output**:
- Saves an interactive Plotly 3D surface plot to disk.

---
