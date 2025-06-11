from .models import (
    train_ols_for_fire as train_ols,
    train_lasso_for_fire as train_lasso,
    train_mlp_for_fire as train_mlp,
    train_xgboost_for_fire as train_xgb
)

from .cnn_module import CNNModel
from .data_utils import (
    load_fire_data_from_excel as load_excel_data,
    apply_scaling
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import pandas as pd
import numpy as np
import argparse


# === Inverse transformation of predictions and targets === #
def inverse_transform_if_needed(arr, scaler, scale_mode):
    arr = arr.reshape(-1, 1)

    if scale_mode == "log_minmax":
        arr = scaler.inverse_transform(arr)
        arr = np.expm1(arr)

    elif scale_mode == "minmax" or scale_mode == "robust":
        arr = scaler.inverse_transform(arr)

    elif scale_mode == "log":
        arr = np.expm1(arr)

    elif scale_mode == "original":
        pass

    else:
        raise ValueError(f"Unsupported scale_mode: {scale_mode}")

    return arr.flatten()



# === Unified metric evaluation in original scale === #
def evaluate_model(y_true_scaled, y_pred_scaled, scaler_y, scale_mode):
    y_true_original = inverse_transform_if_needed(y_true_scaled, scaler_y, scale_mode)
    y_pred_original = inverse_transform_if_needed(y_pred_scaled, scaler_y, scale_mode)

    return {
        "R²": r2_score(y_true_original, y_pred_original),
        "MAE": mean_absolute_error(y_true_original, y_pred_original),
        "MSE": mean_squared_error(y_true_original, y_pred_original)
    }


# === Main training logic for one model === #
def train_fire_model(model_name, data_path, save=True, scale_mode="log_minmax"):
    X, y = load_excel_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = apply_scaling(
        X_train, X_test, y_train, y_test, scale_mode
    )

    if model_name == "cnn":
        model, y_pred_scaled = train_optuna_cnn_for_fire(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
    else:
        if model_name == "ols":
            model = train_ols(X_train_scaled, y_train_scaled)
            X_test_eval = sm.add_constant(X_test_scaled, has_constant="add")
        elif model_name == "lasso":
            model = train_lasso(X_train_scaled, y_train_scaled)
            X_test_eval = X_test_scaled
        elif model_name == "mlp":
            model = train_mlp(X_train_scaled, y_train_scaled)
            X_test_eval = X_test_scaled
        elif model_name == "xgboost":
            model = train_xgb(X_train_scaled, y_train_scaled)
            X_test_eval = X_test_scaled
        else:
            raise ValueError("Unsupported model name")

        y_pred_scaled = model.predict(X_test_eval)

    metrics = evaluate_model(y_test_scaled, y_pred_scaled, scaler_y, scale_mode)

    if save:
        joblib.dump({
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "feature_names": X.columns.tolist(),
            "scale_mode": scale_mode
        }, f"best_{model_name}_model.joblib")

    print(f"\n🔥 Model '{model_name}' Evaluation:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return model, metrics


# === CNN with Optuna tuning === #
def train_optuna_cnn_for_fire(X_train, y_train, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    def objective(trial):
        num_filters1 = trial.suggest_int("num_filters1", 8, 32)
        num_filters2 = trial.suggest_int("num_filters2", 16, 64)
        fc1_size = trial.suggest_int("fc1_size", 32, 128)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        model = CNNModel(num_filters1, num_filters2, fc1_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for _ in range(300):
            model.train()
            optimizer.zero_grad()
            preds = model(X_train_tensor)
            loss = criterion(preds, y_train_tensor)
            if torch.isnan(loss): return float("inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).cpu().numpy()
            if np.isnan(y_pred).any():
                return float("inf")
        return -r2_score(y_test_tensor.cpu().numpy(), y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_params
    model = CNNModel(best_params["num_filters1"], best_params["num_filters2"], best_params["fc1_size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_train_tensor)
        train_loss = criterion(train_pred, y_train_tensor)
        if torch.isnan(train_loss): break
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_tensor)
            val_loss = criterion(val_pred, y_test_tensor)
        if torch.isnan(val_loss): break

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state_dict = model.state_dict()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy().flatten()
        if np.isnan(preds).any():
            print("❌ NaN detected in final predictions!")

    return model, preds




# === CNN Ensemble === #
def train_multiple_cnn_for_fire(data_path, n_runs=5, save=True, scale_mode="log_minmax"):
    X, y = load_excel_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_tr_scaled, X_te_scaled, y_tr_scaled, y_te_scaled, scaler_X, scaler_y = apply_scaling(
        X_train, X_test, y_train, y_test, scale_mode
    )

    models = []
    preds_list = []

    for _ in range(n_runs):
        model, preds_scaled = train_optuna_cnn_for_fire(X_tr_scaled, y_tr_scaled, X_te_scaled, y_te_scaled)
        models.append(model)
        preds_list.append(preds_scaled)

    ensemble_preds_scaled = np.mean(np.stack(preds_list, axis=0), axis=0)

    ensemble_metrics = evaluate_model(y_te_scaled, ensemble_preds_scaled, scaler_y, scale_mode)

    if save:
        joblib.dump({
            "models": models,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "feature_names": X.columns.tolist(),
            "scale_mode": scale_mode
        }, "cnn_ensemble.joblib")

    print("\n🔥 CNN Ensemble Evaluation:")
    for k, v in ensemble_metrics.items():
        print(f"{k}: {v:.4f}")

    return models, preds_list, ensemble_metrics


# === CLI === #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fire risk regression model.")
    parser.add_argument("--model_name", required=True, help="Model to train: ols, lasso, mlp, xgboost, cnn")
    parser.add_argument("--data_path", required=True, help="Path to training Excel file")
    parser.add_argument("--no_save", action="store_true", help="If set, do not save the trained model.")
    parser.add_argument("--scale_mode", choices=["original", "log", "minmax", "log_minmax", "robust"],
                        default="log_minmax", help="Data preprocessing mode")

    args = parser.parse_args()

    train_fire_model(
        model_name=args.model_name,
        data_path=args.data_path,
        save=not args.no_save,
        scale_mode=args.scale_mode
    )