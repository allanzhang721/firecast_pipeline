# 🔥 firecast_pipeline

This repository provides a unified training and prediction pipeline for **fire risk regression tasks** using the following models:

- **OLS** (Ordinary Least Squares)
- **Lasso**
- **MLP** (Multi-layer Perceptron)
- **CNN** (with Optuna hyperparameter tuning)
- **XGBoost**

The pipeline is designed to support `.xlsx` Excel datasets with flexible feature columns.  
👉 **The last column must always be the response (target) variable** (e.g., Time to Flashover).

---

## 📁 Expected Excel Format

- **File type:** `.xlsx`
- **Structure:**
  - ✅ First row = column headers
  - ✅ All columns except the **last** = input features
  - ✅ Last column = fire risk target (e.g., TTF)
  - ❌ Unnecessary columns must be **removed**, not just hidden

### ✅ Example (valid structure)

| Thermal Inertia | HRRPUA | Ignition Temp | Time to Flashover |
|-----------------|--------|----------------|--------------------|
| 136500          | 725    | 400            | 42.5               |
| ...             | ...    | ...            | ...                |

---

## 📦 Installation and Running

Install all required dependencies:

```bash
pip install pandas numpy scikit-learn statsmodels xgboost torch optuna openpyxl
pip install streamlit plotly
python -m regressorpipeline.train --model_name cnn --data_path examples/example_data_train.xlsx
python -m regressorpipeline.predict --predict_path examples/example_data_test.xlsx --model_path examples/best_cnn_model.joblib
python -m regressorpipeline.predict --predict_path examples/example_data_test.xlsx --model_path examples/best_cnn_model.joblib --output_path example/predict_results.csv
python -m regressorpipeline.visualize --feat1 ThermalInertia --feat2 FuelLoadDensity --model_path examples/best_cnn_model.joblib

python -m regressorpipeline.visualize --feat1 ThermalInertia --feat2 FuelLoadDensity --model_path examples/best_cnn_model.joblib --save_path examples/cnn_surface.html
