import argparse
import pandas as pd
import numpy as np
import joblib
from visualize import plot_fire_risk_surface_matplotlib



def main():
    parser = argparse.ArgumentParser(description="Visualize CNN prediction surface for fire risk.")
    parser.add_argument("--feat1", required=True, help="Name of the first feature (X-axis)")
    parser.add_argument("--feat2", required=True, help="Name of the second feature (Y-axis)")
    parser.add_argument("--model_path", required=True, help="Path to best_cnn_model.joblib")
    args = parser.parse_args()

    # Load model bundle
    model_bundle = joblib.load(args.model_path)
    model = model_bundle["model"]
    scaler_X = model_bundle["scaler_X"]
    scaler_y = model_bundle["scaler_y"]
    feature_names = model_bundle["feature_names"]

    # Infer training data from features inside model
    df = pd.read_excel("example_data_train.xlsx").dropna()
    df.columns = [c.strip() for c in df.columns]
    X = df[feature_names]

    # Preprocess
    X_log = np.log1p(X)
    X_scaled = scaler_X.transform(X_log)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    # Plot
    plot_fire_risk_surface_matplotlib(
        model=model,
        X_scaled_df=X_scaled_df,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        feat1_name=args.feat1,
        feat2_name=args.feat2,
        title=f"Fire Risk Surface: {args.feat1} vs {args.feat2}"
    )

if __name__ == "__main__":
    main()
