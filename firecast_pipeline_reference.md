
# 🔥 firecast_pipeline – Updated Function Reference

---

## 📂 Data Utilities

### `load_fire_data_from_excel`

```python
load_fire_data_from_excel(path)
```

Loads an Excel fire dataset and returns features and target. Ensures numeric values and removes formula cells.

**Parameters:**
- `path` (`str`) — Path to `.xlsx` file.

**Returns:**
- `X` (`pd.DataFrame`) — Input features (all columns except last).
- `y` (`pd.Series`) — Target (last column).

---

### `apply_scaling`

```python
apply_scaling(X_train, X_test, y_train, y_test, scale_mode)
```

Scales and transforms feature/target data with support for various modes.

**Parameters:**
- `scale_mode` (`str`) — One of: `"original"`, `"log"`, `"minmax"`, `"log_minmax"`, `"robust"`.

**Returns:**
- `X_train_scaled`, `X_test_scaled`, `y_train_scaled`, `y_test_scaled`, `scaler_X`, `scaler_y`

---

### `log_minmax_scale_fire_data`

```python
log_minmax_scale_fire_data(X_train, X_test, y_train, y_test)
```

Convenience function applying `log1p` and MinMaxScaler to train/test sets.

---

## 🧠 Models

### `train_model_for_fire`

```python
train_model_for_fire(X, y, model_name)
```

Trains a regression model: OLS, Lasso, MLP, or XGBoost.

---

### `CNNModel`

```python
class CNNModel(nn.Module)
```

1D CNN for tabular regression tasks.

**Inputs:**
- `(batch_size, 1, num_features)`  
**Output:**
- `(batch_size, 1)` regression predictions

---

### `train_optuna_cnn_for_fire`

```python
train_optuna_cnn_for_fire(X_train, y_train, X_test, y_test)
```

Trains a CNN with Optuna hyperparameter search.

**Returns:**
- Trained model (`CNNModel`)
- `np.ndarray` — Scaled predictions on test set

---

### `train_fire_model`

```python
train_fire_model(model_name, data_path, save=True, scale_mode="log_minmax")
```

Unified CLI-compatible training function.

---

### `train_multiple_cnn_for_fire`

```python
train_multiple_cnn_for_fire(data_path, n_runs=5, save=True, scale_mode="log_minmax")
```

Runs `n_runs` of CNN training for ensemble modeling. Averages predictions.

**Returns:**
- `models` — list of trained `CNNModel`
- `preds_list` — list of individual prediction arrays
- `ensemble_metrics` — aggregated metrics on averaged predictions

---

## 🔮 Prediction

### `load_model_bundle`

```python
load_model_bundle(model_path)
```

Loads a `.joblib` model, scalers, and metadata (feature names, scale mode).

---

### `predict_fire_risk`

```python
predict_fire_risk(model, scaler_X, scaler_y, input_path, ...)
```

Runs prediction using a single model.

**Returns:**
- `np.ndarray` — Original-scale predictions

---

### `predict_fire_risk_from_models`

```python
predict_fire_risk_from_models(models, scaler_X, scaler_y, input_path, ...)
```

Averages predictions from multiple models.

---

## 📈 Visualization

### `plot_fire_risk_surface_matplotlib`

```python
plot_fire_risk_surface_matplotlib(
    model, scaler_X, scaler_y, feat1_name, feat2_name, data_path, scale_mode, ...
)
```

Generates a 3D surface plot of predictions over 2 features using Plotly.

**Saves:**
- Interactive `.html` surface plot

---

### `plot_predictions_with_nan`

```python
plot_predictions_with_nan(true_values, predicted_values, title, save_path=None)
```

Plots predicted vs. ground truth values. Highlights any sample where:
- Prediction is `NaN`
- Ground truth is `NaN`
- Both are `NaN`
