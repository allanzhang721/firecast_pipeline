import argparse
import pandas as pd
import numpy as np
import joblib
import torch



def predict_fire_risk_from_model(model_path, input_path):
    # Load model bundle
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    scaler_X = model_bundle["scaler_X"]
    scaler_y = model_bundle["scaler_y"]

    # Load test features from Excel
    df = pd.read_excel(input_path, engine="openpyxl")
    X = np.log1p(df.select_dtypes(include=[np.number]))  # Apply log1p transform
    X_scaled = scaler_X.transform(X)

    # Predict
    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)).numpy()
    else:
        preds = model.predict(X_scaled)

    # Inverse transform
    preds = np.expm1(scaler_y.inverse_transform(preds.reshape(-1, 1))).ravel()
    return preds

def predict_fire_risk_from_multiple_models(model_paths, input_path):
    """Run prediction using several saved models and average the results.

    Parameters
    ----------
    model_paths : list of str
        Paths to `.joblib` model bundles.
    input_path : str
        Path to the Excel file containing samples to predict.

    Returns
    -------
    np.ndarray
        Averaged predictions from all provided models.
    """

    if not model_paths:
        raise ValueError("model_paths list cannot be empty")

    # Load input data once
    df = pd.read_excel(input_path, engine="openpyxl")
    X_num = np.log1p(df.select_dtypes(include=[np.number]))

    preds_list = []

    for p in model_paths:
        bundle = joblib.load(p)
        if "model" in bundle:
            models = [bundle["model"]]
        else:
            models = bundle.get("models", [])
        scaler_X = bundle["scaler_X"]
        scaler_y = bundle["scaler_y"]

        X_scaled = scaler_X.transform(X_num)

        combined_pred = []
        for m in models:
            if isinstance(m, torch.nn.Module):
                m.eval()
                with torch.no_grad():
                    pr = m(torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)).numpy()
            else:
                pr = m.predict(X_scaled)
            combined_pred.append(pr)

        avg_pred = np.mean(np.stack(combined_pred, axis=0), axis=0)
        avg_pred = np.expm1(scaler_y.inverse_transform(avg_pred.reshape(-1, 1))).ravel()
        preds_list.append(avg_pred)

    return np.mean(np.stack(preds_list, axis=0), axis=0)


def main():
    parser = argparse.ArgumentParser(description="Run fire risk prediction using trained model.")
    parser.add_argument("--predict_path", required=True, help="Path to .xlsx file with test features")
    parser.add_argument("--model_path", required=True, help="Path to .joblib trained model file")
    parser.add_argument("--output_path", default=None, help="Optional: path to save predictions as CSV")
    args = parser.parse_args()

    # Run prediction
    preds = predict_fire_risk_from_model(args.model_path, args.predict_path)

    print("\n🔥 Fire Risk Predictions:")
    print(preds)

    # Optionally save results
    if args.output_path:
        df_out = pd.DataFrame({"Predicted Fire Risk": preds})
        df_out.to_csv(args.output_path, index=False)
        print(f"\n✅ Predictions saved to {args.output_path}")


if __name__ == "__main__":
    main()
