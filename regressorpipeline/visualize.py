import argparse
import pandas as pd
import numpy as np
import joblib
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def plot_fire_risk_surface_matplotlib(
    model,
    scaler_X,
    scaler_y,
    feat1_name,
    feat2_name,
    data_path,
    scale_mode="original",
    feature_names=None,
    title="CNN Prediction Surface",
    save_path="fire_risk_surface.html"
):
    import pandas as pd
    import numpy as np
    import torch
    import plotly.graph_objects as go

    # Load and clean data
    df = pd.read_excel(data_path, engine="openpyxl").dropna()
    df.columns = [c.strip() for c in df.columns]
    
    if feature_names is None:
        feature_names = df.select_dtypes(include="number").columns.tolist()

    X = df[feature_names]

    # Apply log1p if needed
    if scale_mode in ["log", "log_minmax"]:
        X = np.log1p(X)

    # Apply scaler if provided
    if scaler_X is not None:
        X_scaled = scaler_X.transform(X)
    else:
        X_scaled = X.values

    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    # === Begin surface logic ===
    idx1 = feature_names.index(feat1_name)
    idx2 = feature_names.index(feat2_name)

    f1_vals = np.linspace(X_scaled_df.iloc[:, idx1].min(), X_scaled_df.iloc[:, idx1].max(), 30)
    f2_vals = np.linspace(X_scaled_df.iloc[:, idx2].min(), X_scaled_df.iloc[:, idx2].max(), 30)
    F1, F2 = np.meshgrid(f1_vals, f2_vals)

    X_grid_full_scaled = np.tile(X_scaled_df.mean().values, (F1.size, 1))
    X_grid_full_scaled[:, idx1] = F1.ravel()
    X_grid_full_scaled[:, idx2] = F2.ravel()

    with torch.no_grad():
        Y_pred = model(torch.tensor(X_grid_full_scaled, dtype=torch.float32).unsqueeze(1)).numpy()

    # Inverse transform features
    if scaler_X is not None:
        X_grid_log = scaler_X.inverse_transform(X_grid_full_scaled)
        X_grid_original = np.expm1(X_grid_log) if scale_mode in ["log", "log_minmax"] else X_grid_log
    else:
        X_grid_original = np.expm1(X_grid_full_scaled) if scale_mode in ["log", "log_minmax"] else X_grid_full_scaled

    F1_original = X_grid_original[:, idx1].reshape(F1.shape)
    F2_original = X_grid_original[:, idx2].reshape(F2.shape)

    # Inverse transform predictions
    if scaler_y is not None:
        y_grid_pred_log = scaler_y.inverse_transform(Y_pred.reshape(-1, 1)).ravel()
        y_grid_pred_original = np.expm1(y_grid_pred_log).reshape(F1.shape)
    else:
        y_grid_pred_original = Y_pred.reshape(F1.shape)

    # Plot
    fig = go.Figure(data=[go.Surface(
        z=y_grid_pred_original,
        x=F1_original,
        y=F2_original,
        colorscale='Viridis'
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=feat1_name,
            yaxis_title=feat2_name,
            zaxis_title="Predicted Fire Risk"
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    fig.write_html(save_path)
    print(f"✅ 3D surface plot saved to: {save_path}")


import numpy as np
import matplotlib.pyplot as plt

def plot_predictions_with_nan(true_values, predicted_values, title="Predicted vs. Ground Truth", save_path=None):
    """
    Plot predicted vs. ground truth values while marking samples where either value is NaN.

    Parameters:
    - true_values: list or np.ndarray with possible NaNs
    - predicted_values: list or np.ndarray with possible NaNs
    - title: optional plot title
    - save_path: if provided, saves the plot to this path
    """
    true_values = np.array(true_values, dtype=np.float32)
    predicted_values = np.array(predicted_values, dtype=np.float32)
    x = np.arange(len(true_values))

    # Identify valid and NaN points
    valid_mask = ~np.isnan(true_values) & ~np.isnan(predicted_values)
    nan_mask = np.isnan(true_values) | np.isnan(predicted_values)

    plt.figure(figsize=(8, 5))

    # Plot valid points
    plt.plot(x[valid_mask], predicted_values[valid_mask], 'x--', label='Predicted', color='tab:orange')
    plt.plot(x[valid_mask], true_values[valid_mask], 'o-', label='Ground Truth', color='tab:blue')

    # Mark NaN points
    for i in x[nan_mask]:
        y_pred = predicted_values[i]
        y_true = true_values[i]

        if np.isnan(y_true) and not np.isnan(y_pred):
            plt.scatter(i, y_pred, color='red', marker='x', label='Predicted (no GT)' if i == x[nan_mask][0] else "")
            plt.text(i, y_pred, "No GT", fontsize=8, ha='center', va='bottom', color='red')
        elif np.isnan(y_pred) and not np.isnan(y_true):
            plt.scatter(i, y_true, color='purple', marker='s', label='GT (no Pred)' if i == x[nan_mask][0] else "")
            plt.text(i, y_true, "No Pred", fontsize=8, ha='center', va='bottom', color='purple')
        elif np.isnan(y_true) and np.isnan(y_pred):
            plt.scatter(i, 0, color='black', marker='|', label='No GT/Pred' if i == x[nan_mask][0] else "")
            plt.text(i, 0, "No GT/Pred", fontsize=8, ha='center', va='bottom', color='black')

    plt.xlabel("Sample Index")
    plt.ylabel("Fire Risk")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Saved plot to {save_path}")
    else:
        plt.show()



def main():
    parser = argparse.ArgumentParser(description="Fire risk model visualization utilities.")
    parser.add_argument("--feat1", required=True)
    parser.add_argument("--feat2", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--scale_mode", choices=["original", "log", "minmax", "log_minmax"], default=None)
    parser.add_argument("--save_path", default="fire_risk_surface.html")

    args = parser.parse_args()

    # Load model bundle
    model_bundle = joblib.load(args.model_path)
    model = model_bundle["model"]
    scaler_X = model_bundle.get("scaler_X", None)
    scaler_y = model_bundle.get("scaler_y", None)
    feature_names = model_bundle["feature_names"]
    scale_mode = args.scale_mode or model_bundle.get("scale_mode", "original")

    # Generate surface plot
    plot_fire_risk_surface_matplotlib(
        model=model,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        feature_names=feature_names,
        feat1_name=args.feat1,
        feat2_name=args.feat2,
        data_path=args.data_path,
        scale_mode=scale_mode,
        title=f"Fire Risk Surface: {args.feat1} vs {args.feat2}",
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()