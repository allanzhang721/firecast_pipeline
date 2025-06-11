import argparse
import pandas as pd
import numpy as np
import joblib
import torch
import statsmodels.api as sm


def load_model_bundle(model_path):
    bundle = joblib.load(model_path)
    models = bundle.get("models") or [bundle["model"]]
    scaler_X = bundle.get("scaler_X", None)
    scaler_y = bundle.get("scaler_y", None)
    feature_names = bundle.get("feature_names", None)

    return models, scaler_X, scaler_y, feature_names


def preprocess_input(input_path, scaler_X, feature_names):
    df = pd.read_excel(input_path, engine="openpyxl")
    X = df.select_dtypes(include=[np.number])

    if feature_names is not None:
        missing = set(feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features in input: {missing}")
        X = X[feature_names]
    elif scaler_X is not None and hasattr(scaler_X, "feature_names_in_"):
        feature_names_in = scaler_X.feature_names_in_
        missing = set(feature_names_in) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features in input: {missing}")
        X = X[feature_names_in]

    X_scaled = scaler_X.transform(X) if scaler_X is not None else X.values
    return X_scaled


def inverse_transform_predictions(preds, scaler_y, scale_mode):
    if scaler_y is not None:
        preds = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
    if scale_mode in ["log", "log_minmax"]:
        preds = np.expm1(preds)
    return preds


def predict_fire_risk(model, scaler_X, scaler_y, input_path, feature_names=None, model_type=None, scale_mode="original"):
    # Step 1: Load and scale input
    X_scaled = preprocess_input(input_path, scaler_X, feature_names)
    
    # Step 2: Check for NaNs in input
    if np.isnan(X_scaled).any():
        print("❌ NaN detected in model input (X_scaled). Please check the input Excel file or preprocessing.")
        print("Problematic input rows:\n", X_scaled[np.isnan(X_scaled).any(axis=1)])
    
    # Step 3: Run inference
    if isinstance(model, torch.nn.Module) or model_type == "cnn":
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
            preds = model(X_tensor).numpy().flatten()

        # Step 4: Check for NaNs in raw predictions
        if np.isnan(preds).any():
            print("❌ NaN detected in raw CNN predictions!")
            print("Raw preds:", preds)

    elif isinstance(model, sm.regression.linear_model.RegressionResultsWrapper) or model_type == "ols":
        X_eval = sm.add_constant(X_scaled, has_constant="add")
        preds = model.predict(X_eval)

    else:  # sklearn
        preds = model.predict(X_scaled)

    # Step 5: Apply inverse transform
    final_preds = inverse_transform_predictions(preds, scaler_y, scale_mode)

    # Step 6: Check for NaNs in final predictions
    if np.isnan(final_preds).any():
        print("❌ NaN detected in final predictions after inverse transform!")
        print("Final preds:", final_preds)

    return final_preds



def predict_fire_risk_from_models(models, scaler_X, scaler_y, input_path, feature_names=None, model_type=None, scale_mode="original"):
    X_scaled = preprocess_input(input_path, scaler_X, feature_names)

    preds_list = []
    for m in models:
        x_input = X_scaled.copy()

        if isinstance(m, torch.nn.Module) or model_type == "cnn":
            m.eval()
            with torch.no_grad():
                pred = m(torch.tensor(x_input, dtype=torch.float32).unsqueeze(1)).numpy().flatten()
        elif isinstance(m, sm.regression.linear_model.RegressionResultsWrapper) or model_type == "ols":
            x_input = sm.add_constant(x_input, has_constant="add")
            pred = m.predict(x_input)
        else:
            pred = m.predict(x_input)

        preds_list.append(pred)

    avg_pred = np.mean(np.stack(preds_list, axis=0), axis=0)
    return inverse_transform_predictions(avg_pred, scaler_y, scale_mode)


def main():
    parser = argparse.ArgumentParser(description="Run fire risk prediction using trained model.")
    parser.add_argument("--predict_path", required=True, help="Path to .xlsx file with test features")
    parser.add_argument("--model_path", required=True, help="Path to .joblib trained model file")
    parser.add_argument("--model_type", default=None, help="Optional: specify model type: ols, cnn, xgboost, etc.")
    parser.add_argument("--output_path", default=None, help="Optional: path to save predictions as CSV")
    args = parser.parse_args()

    models, scaler_X, scaler_y, feature_names, scale_mode = load_model_bundle(args.model_path)

    if len(models) == 1:
        preds = predict_fire_risk(
            models[0], scaler_X, scaler_y, args.predict_path,
            feature_names=feature_names, model_type=args.model_type,
            scale_mode=scale_mode
        )
    else:
        preds = predict_fire_risk_from_models(
            models, scaler_X, scaler_y, args.predict_path,
            feature_names=feature_names, model_type=args.model_type,
            scale_mode=scale_mode
        )

    print("\n🔥 Fire Risk Predictions:")
    print(preds)

    if args.output_path:
        df_out = pd.DataFrame({"Predicted Fire Risk": preds})
        df_out.to_csv(args.output_path, index=False)
        print(f"\n✅ Predictions saved to {args.output_path}")


if __name__ == "__main__":
    main()
