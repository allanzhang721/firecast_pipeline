
# 🔥 firecast_pipeline – Updated Function Reference

---

## `load_fire_data_from_excel`

```python
load_fire_data_from_excel(path)
```

Load a fire dataset from an Excel file and split it into features and target.

### Parameters:
- `path` (`str`) — Path to the `.xlsx` file.

### Returns:
- `X` (`pd.DataFrame`) — Input features (all columns except the last).
- `y` (`pd.Series`) — Target variable (last column).

---

## `log_minmax_scale_fire_data`

```python
log_minmax_scale_fire_data(X_train, X_test, y_train, y_test)
```

Apply `log1p` transformation followed by MinMax scaling to fire data.

### Parameters:
- `X_train`, `X_test` (`pd.DataFrame`) — Feature matrices.
- `y_train`, `y_test` (`pd.Series`) — Target variables.

### Returns:
- `X_train_scaled`, `X_test_scaled`, `y_train_scaled`, `y_test_scaled` (`np.ndarray`)
- `scaler_X`, `scaler_y` — Fitted `MinMaxScaler` instances.

---

## `train_model_for_fire`

```python
train_model_for_fire(X, y, model_name)
```

Train a regression model for fire data using the specified model type.

### Parameters:
- `X` (`pd.DataFrame`) — Input features.
- `y` (`pd.Series`) — Target variable.
- `model_name` (`str`) — One of `ols`, `lasso`, `mlp`, `xgboost`.

### Returns:
- Trained model instance (e.g., `LinearRegression`, `Lasso`, `MLPRegressor`, `XGBRegressor`).

---

## `CNNModel`

```python
class CNNModel(nn.Module)
```

A 1D CNN model for regression tasks on tabular data.

### Constructor Parameters:
- `num_filters1` (`int`) — Filters in the first conv layer.
- `num_filters2` (`int`) — Filters in the second conv layer.
- `fc1_size` (`int`) — Hidden units in the fully connected layer.

### Input:
- Tensor of shape `(batch_size, 1, num_features)`

### Returns:
- Tensor of shape `(batch_size, 1)` — Regression prediction.

---

## `train_optuna_cnn_for_fire`

```python
train_optuna_cnn_for_fire(X_train, y_train, X_test, y_test)
```

Train a CNN model using Optuna for hyperparameter tuning.

### Parameters:
- `X_train`, `X_test` — Scaled feature arrays (`np.ndarray`)
- `y_train`, `y_test` — Scaled target arrays (`np.ndarray`)

### Returns:
- `best_model` — Trained PyTorch model.
- `metrics` (`dict`) — Evaluation metrics: R², MAE, MSE.

---

## `train_multiple_cnn_for_fire`

```python
train_multiple_cnn_for_fire(data_path, n_runs=5, save=True)
```

Train a CNN ensemble by running `n_runs` trainings on the same dataset and averaging the predictions.

### Parameters:
- `data_path` (`str`) — Path to the Excel dataset.
- `n_runs` (`int`, optional) — Number of training runs.
- `save` (`bool`, optional) — Whether to save the ensemble as cnn_ensemble.joblib in my current directory. Default is True.

### Returns:
- `list` — Trained CNN models.
- `list[dict]` — Metrics for each run.
- `dict` — Metrics computed from the averaged ensemble predictions.

---

## `predict_fire_risk`

```python
predict_fire_risk(model, scaler_X, scaler_y, input_path)
```

Predict fire risk values from a single loaded model and scaler.

### Parameters:
- `model` — Trained model (torch or sklearn).
- `scaler_X`, `scaler_y` — Fitted `MinMaxScaler` instances.
- `input_path` (`str`) — Path to `.xlsx` file with test features.

### Returns:
- `np.ndarray` — Predictions in original units.

---

## `predict_fire_risk_from_models`

```python
predict_fire_risk_from_models(models, scaler_X, scaler_y, input_path)
```

Predict using several models and return the average prediction.

### Parameters:
- `models` (`list`) — List of trained models (e.g., PyTorch, sklearn).
- `scaler_X`, `scaler_y` — Fitted `MinMaxScaler` instances.
- `input_path` (`str`) — Path to `.xlsx` file with test features.

### Returns:
- `np.ndarray` — Averaged predictions in original units.

---

## `load_model_bundle`

```python
load_model_bundle(model_path)
```

Load a saved model bundle from a `.joblib` file.

### Parameters:
- `model_path` (`str`) — Path to saved `.joblib` file.

### Returns:
- `models` (`list`) — One or more trained models.
- `scaler_X`, `scaler_y` — Corresponding scalers.

---

## `plot_fire_risk_surface_matplotlib`

```python
plot_fire_risk_surface_matplotlib(model, X_scaled_df, scaler_X, scaler_y, feat1_name, feat2_name, title, save_path="fire_risk_surface.html")
```

Plot and save a 3D prediction surface using Plotly based on two input features.

### Parameters:
- `model` — Trained PyTorch CNN model.
- `X_scaled_df` (`pd.DataFrame`) — Scaled + log-transformed features.
- `scaler_X`, `scaler_y` — Scalers used during training.
- `feat1_name`, `feat2_name` (`str`) — Feature names to plot on the X and Y axes.
- `title` (`str`) — Plot title.
- `save_path` (`str`) — File path to save the interactive `.html` surface plot.

### Output:
- Saves an interactive Plotly 3D surface plot to disk.
