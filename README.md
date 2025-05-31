# 🔥 firecast_pipeline

This repository provides a unified training and prediction pipeline for **fire risk regression tasks** using the following models:

- **OLS** (Ordinary Least Squares)
- **Lasso**
- **MLP** (Multi-layer Perceptron)
- **CNN** (with Optuna hyperparameter tuning)
- **XGBoost**

The pipeline is designed for `.xlsx` Excel datasets with flexible feature columns.

👉 **The last column must always be the response (target) variable** (e.g., Time to Flashover).

---

## 📁 Expected Excel Format

- **File type:** `.xlsx`
- **Structure:**
  - ✅ First row = column headers
  - ✅ All columns except the **last** = input features
  - ✅ Last column = fire risk target (e.g., TTF)
  - ❌ Unnecessary columns must be **removed**, not just hidden

### ✅ Example

| Thermal Inertia | HRRPUA | Ignition Temp | Time to Flashover |
|-----------------|--------|----------------|--------------------|
| 136500          | 725    | 400            | 42.5               |
| ...             | ...    | ...            | ...                |

---

## 📦 Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

> Or manually:

```bash
pip install pandas numpy scikit-learn statsmodels xgboost torch optuna openpyxl joblib plotly
```

---

## 🚀 Training

Train any supported model directly from Python:

```python
from regressorpipeline.train import train_fire_model, train_multiple_cnn_for_fire

# Train a single model
train_fire_model("cnn", "examples/example_data_train.xlsx")

# Train multiple CNNs on separate time series and average metrics
models, metrics_list, avg_metrics = train_multiple_cnn_for_fire([
    "examples/example_data_train.xlsx",
    "examples/another_time_series.xlsx",
])
```

Trained models are saved to the `examples/` folder as `best_<model_name>_model.joblib`.


### Train a CNN Ensemble

You can train the same dataset several times to build a more stable ensemble:

```python
from regressorpipeline.train import train_multiple_cnn_for_fire

models, run_metrics, ensemble_metrics = train_multiple_cnn_for_fire(
    "examples/example_data_train.xlsx", n_runs=3
)
print(ensemble_metrics)
```

The trained ensemble is saved as `examples/cnn_ensemble.joblib`.

---

## 🔍 Prediction

Use the prediction utilities from Python code:

```python
from regressorpipeline.predict import (
    predict_fire_risk_from_model,
    predict_fire_risk_from_multiple_models,
)

# Predict with a single trained model
preds = predict_fire_risk_from_model(
    "examples/best_cnn_model.joblib",
    "examples/example_data_test.xlsx",
)

# Predict with multiple models and average the results
pred_lists, avg_preds = predict_fire_risk_from_multiple_models(
    ["examples/best_cnn_model.joblib", "examples/best_ols_model.joblib"],
    ["examples/example_data_test.xlsx", "examples/example_data_test.xlsx"],
)
```

---

## 📊 Visualization (CNN only)

Generate a 3D surface plot for CNN predictions over any two features:

```python
from regressorpipeline.visualize import plot_fire_risk_surface_matplotlib

plot_fire_risk_surface_matplotlib(
    model,
    X_scaled_df,
    scaler_X,
    scaler_y,
    "ThermalInertia",
    "FuelLoadDensity",
    "CNN Surface",
    save_path="examples/cnn_surface.html",
)
```

The plot is saved as an interactive HTML file.
---

## 📂 Folder Structure

```text
firecast_pipeline/
│
├── regressorpipeline/
│   ├── train.py                # Training logic
│   ├── predict.py              # Prediction logic
│   ├── visualize.py            # 3D surface visualization
│   ├── cnn_module.py           # CNN model definition
│   ├── models.py               # Traditional model trainers
│   └── data_utils.py           # Data loaders and scalers
│
├── examples/
│   ├── example_data_train.xlsx
│   ├── example_data_test.xlsx
│   └── best_cnn_model.joblib
│
├── requirements.txt
└── README.md
```

---

## 📜 License

MIT License – use freely for research or fire safety AI applications. For commercial use, please contact the authors.
