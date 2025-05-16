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

Apply log transformation and MinMax scaling to fire data.

**Parameters**:
- `X_train`, `X_test` (`pd.DataFrame`) — Feature matrices.
- `y_train`, `y_test` (`pd.Series`) — Target variables.

**Returns**:
- `X_train_scaled`, `X_test_scaled`, `y_train_scaled`, `y_test_scaled` — Scaled NumPy arrays.
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
- Fitted model instance.

---

## `CNNModel`

```python
class CNNModel(nn.Module)
```

A 1D CNN model for regression tasks.

**Constructor Parameters**:
- `num_filters1`, `num_filters2` — Number of convolutional filters.
- `fc1_size` — Number of units in the fully connected layer.

**Forward Input**:
- Tensor of shape `(batch, 1, features)`

**Returns**:
- Tensor of shape `(batch, 1)`

---

## `train_optuna_cnn_for_fire`

```python
train_optuna_cnn_for_fire(X_train, y_train, X_test, y_test)
```

Train a CNN model with hyperparameter tuning using Optuna.

**Returns**:
- Best CNN model found.

---

## `predict_fire_risk_from_model`

```python
predict_fire_risk_from_model(model_path, input_path, model_name)
```

Predict fire risk values using a saved `.joblib` model and specified model type.

**Parameters**:
- `model_path` (`str`) — Path to trained model.
- `input_path` (`str`) — Path to `.xlsx` file with test features.
- `model_name` (`str`) — One of `ols`, `lasso`, `mlp`, `xgboost`, `cnn`.

**Returns**:
- `np.ndarray` — Predictions in original units (after inverse transform).

---

## `plot_fire_risk_surface_matplotlib`

```python
plot_fire_risk_surface_matplotlib(model, X_scaled_df, scaler_X, scaler_y, feat1_name, feat2_name, title)
```

Plot a 3D prediction surface over two features using a trained CNN model.

**Parameters**:
- `model` — Trained PyTorch model.
- `X_scaled_df` — Feature matrix (log-scaled + normalized).
- `scaler_X`, `scaler_y` — Fitted scalers.
- `feat1_name`, `feat2_name` (`str`) — Names of the two features.
- `title` (`str`) — Plot title.

**Displays**:
- A `matplotlib` 3D surface plot.