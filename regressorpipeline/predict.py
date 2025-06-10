import argparse
import pandas as pd
import numpy as np
import joblib
import torch
import statsmodels.api as sm
from sklearn.base import BaseEstimator


def predict_fire_risk(model, scaler_X, scaler_y, input_path, feature_names=None, model_type=None):
    df = pd.read_excel(input_path, engine="openpyxl")
    X = np.log1p(df.select_dtypes(include=[np.number]))

    # Ensure feature alignment
    if feature_names is not None:
        X = X[feature_names]
    elif hasattr(scaler_X, "feature_names_in_"):
        X = X[scaler_X.feature_names_in_]

    X_scaled = scaler_X.transform(X)

    # PyTorch CNN
    if isinstance(model, torch.nn.Module) or model_type == "cnn":
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)).numpy()

    # statsmodels OLS
    elif isinstance(model, sm.regression.linear_model.RegressionResultsWrapper) or model_type == "ols":
        X_scaled = sm.add_constant(X_scaled, has_constant="add")
        preds = model.predict(X_scaled)

    # sklearn model
    else:
        preds = model.predict(X_scaled)

    preds = np.expm1(scaler_y.inverse_transform(preds.reshape(-1, 1))).ravel()
    return preds


def predict_fire_risk_from_models(models, scaler_X, scaler_y, input_path, feature_names=None, model_type=None):
    df = pd.read_excel(input_path, engine="openpyxl")
    X = np.log1p(df.select_dtypes(include=[np.number]))

    if feature_names is not None:
        X = X[feature_names]
    elif hasattr(scaler_X, "feature_names_in_"):
        X = X[scaler_X.feature_names_in_]

    X_scaled_raw = scaler_X.transform(X)
    preds_list = []

    for m in models:
        X_scaled = X_scaled_raw.copy()

        if isinstance(m, torch.nn.Module) or model_type == "cnn":
            m.eval()
            with torch.no_grad():
                pred = m(torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)).numpy()
        elif isinstance(m, sm.regression.linear_model.RegressionResultsWrapper) or model_type == "ols":
            X_scaled = sm.add_constant(X_scaled, has_constant="add")
            pred = m.predict(X_scaled)
        else:
            pred = m.predict(X_scaled)

        preds_list.append(pred)

    avg_pred = np.mean(np.stack(preds_list, axis=0), axis=0)
    avg_pred = np.expm1(scaler_y.inverse_transform(avg_pred.reshape(-1, 1))).ravel()
    return avg_pred

def load_model_bundle(model_path):
    bundle = joblib.load(model_path)
    models = bundle.get("models") or [bundle["model"]]
    scaler_X = bundle["scaler_X"]
    scaler_y = bundle["scaler_y"]
    feature_names = bundle.get("feature_names", None)
    return models, scaler_X, scaler_y, feature_names



def main():
    parser = argparse.ArgumentParser(description="Run fire risk prediction using trained model.")
    parser.add_argument("--predict_path", required=True, help="Path to .xlsx file with test features")
    parser.add_argument("--model_path", required=True, help="Path to .joblib trained model file")
    parser.add_argument("--model_type", default=None, help="Optional: specify model type: ols, cnn, xgboost, etc.")
    parser.add_argument("--output_path", default=None, help="Optional: path to save predictions as CSV")
    args = parser.parse_args()

    models, scaler_X, scaler_y, feature_names = load_model_bundle(args.model_path)

    if len(models) == 1:
        preds = predict_fire_risk(models[0], scaler_X, scaler_y, args.predict_path,
                                  feature_names=feature_names, model_type=args.model_type)
    else:
        preds = predict_fire_risk_from_models(models, scaler_X, scaler_y, args.predict_path,
                                              feature_names=feature_names, model_type=args.model_type)

    print("\n🔥 Fire Risk Predictions:")
    print(preds)

    if args.output_path:
        df_out = pd.DataFrame({"Predicted Fire Risk": preds})
        df_out.to_csv(args.output_path, index=False)
        print(f"\n✅ Predictions saved to {args.output_path}")


if __name__ == "__main__":
    main()
