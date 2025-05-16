"""
Functions:
- predict_fire_risk_from_model(model_path, input_path): Loads model and scalers from .joblib and returns fire risk predictions from Excel input.
"""

import joblib
import numpy as np
import pandas as pd
import torch

def predict_fire_risk_from_model(model_path, input_path):
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    scaler_X = model_bundle["scaler_X"]
    scaler_y = model_bundle["scaler_y"]

    df = pd.read_excel(input_path, engine="openpyxl")
    X = np.log1p(df.select_dtypes(include=[np.number]))
    X_scaled = scaler_X.transform(X)

    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)).numpy()
    else:
        preds = model.predict(X_scaled)

    preds = np.expm1(scaler_y.inverse_transform(preds.reshape(-1, 1))).ravel()
    return preds