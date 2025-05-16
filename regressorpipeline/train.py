"""
Functions:
- train_fire_model(model_name, data_path): Entry point to train a model (ols, lasso, mlp, xgboost, cnn) for fire prediction.
- train_optuna_cnn_for_fire(X_train, y_train, X_test, y_test): Train CNN with Optuna tuning for fire hazard regression.
"""

from .models import train_ols, train_lasso, train_mlp, train_xgb
from .cnn_module import CNNModel
from .data_utils import load_excel_data, log_scale_transform
from sklearn.model_selection import train_test_split
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

def train_fire_model(model_name, data_path):
    X, y = load_excel_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = log_scale_transform(X_train, X_test, y_train, y_test)

    if model_name == "ols":
        model = train_ols(X_train_scaled, y_train_scaled)
    elif model_name == "lasso":
        model = train_lasso(X_train_scaled, y_train_scaled)
    elif model_name == "mlp":
        model = train_mlp(X_train_scaled, y_train_scaled)
    elif model_name == "xgboost":
        model = train_xgb(X_train_scaled, y_train_scaled)
    elif model_name == "cnn":
        model = train_optuna_cnn_for_fire(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
    else:
        raise ValueError("Unsupported model name")

    joblib.dump({"model": model, "scaler_X": scaler_X, "scaler_y": scaler_y}, f"{model_name}_model.joblib")

def train_optuna_cnn_for_fire(X_train, y_train, X_test, y_test):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    def objective(trial):
        model = CNNModel(
            trial.suggest_int("num_filters1", 8, 32),
            trial.suggest_int("num_filters2", 16, 64),
            trial.suggest_int("fc1_size", 32, 128)
        )
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for _ in range(100):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_train_tensor), y_train_tensor)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(X_test_tensor).detach().numpy()
        return -((y_test_tensor.numpy() - pred) ** 2).mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    best_model = CNNModel(best_params["num_filters1"], best_params["num_filters2"], best_params["fc1_size"])
    optimizer = optim.Adam(best_model.parameters(), lr=best_params["lr"])
    criterion = nn.MSELoss()
    for _ in range(100):
        best_model.train()
        optimizer.zero_grad()
        loss = criterion(best_model(X_train_tensor), y_train_tensor)
        loss.backward()
        optimizer.step()
    return best_model