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
python train_predict_pipeline.py --data_path example_data_train.xlsx
python train_predict_pipeline.py --data_path example_data_train.xlsx --model cnn
python train_predict_pipeline.py --predict_path example_data_test.xlsx --model_path best_cnn_model.joblib
python run_visualize.py