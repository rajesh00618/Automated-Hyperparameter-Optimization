import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score

def objective(trial, X_train, y_train):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "random_state": 42,
        "objective": "reg:squarederror",
        "n_jobs": 1
    }

    model = xgb.XGBRegressor(**params)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    cv_mse = -scores.mean()

    return -cv_mse
