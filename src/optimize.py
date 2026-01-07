import time
import json
import random
import numpy as np
import optuna
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, r2_score
from data_loader import load_and_split_data
from objective import objective
from evaluate import evaluate_best_model, save_results
import xgboost as xgb

# -------------------------
# Global Seed Control
# -------------------------
random.seed(42)
np.random.seed(42)

# -------------------------
# MLflow Setup
# -------------------------
mlflow.set_tracking_uri("file:///../outputs/mlruns")
mlflow.set_experiment("optuna-xgboost-optimization")

# -------------------------
# Load Data
# -------------------------
X_train, X_test, y_train, y_test = load_and_split_data()

# -------------------------
# Optuna Study
# -------------------------
study = optuna.create_study(
    study_name="xgboost-housing-optimization",
    direction="maximize",
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    storage="sqlite:///../outputs/optuna_study.db",
    load_if_exists=True
)

start_time = time.time()

study.optimize(
    lambda trial: objective(trial, X_train, y_train),
    n_trials=100,
    n_jobs=2,
)

optimization_time = time.time() - start_time

# -------------------------
# Log all trials to MLflow after optimization
# -------------------------
for trial in study.trials:
    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        # Log hyperparameters
        params = trial.params
        mlflow.log_params(params)

        # Calculate metrics
        cv_mse = -trial.value  # Since we return -cv_mse and maximize
        cv_rmse = np.sqrt(cv_mse)

        # Log metrics
        mlflow.log_metric("cv_mse", cv_mse)
        mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.log_metric("trial_number", trial.number)

        # Tag with trial state
        state_tag = {
            optuna.trial.TrialState.COMPLETE: "COMPLETE",
            optuna.trial.TrialState.PRUNED: "PRUNED",
            optuna.trial.TrialState.FAIL: "FAIL"
        }.get(trial.state, "UNKNOWN")
        mlflow.set_tag("trial_state", state_tag)

# -------------------------
# Best Model Training and Evaluation
# -------------------------
results, best_model = evaluate_best_model(
    study, X_train, X_test, y_train, y_test, optimization_time
)

with mlflow.start_run(run_name="best_model"):
    mlflow.set_tag("best_model", "true")
    mlflow.log_params(results["best_params"])
    mlflow.log_metric("test_mse", results["test_rmse"]**2)  # Convert RMSE back to MSE
    mlflow.log_metric("test_rmse", results["test_rmse"])
    mlflow.log_metric("test_r2", results["test_r2"])
    mlflow.xgboost.log_model(best_model, artifact_path="model")

# -------------------------
# Visualizations
# -------------------------
from optuna.visualization import plot_optimization_history, plot_param_importances

fig1 = plot_optimization_history(study)
fig1.write_image("../outputs/optimization_history.png")

fig2 = plot_param_importances(study)
fig2.write_image("../outputs/param_importance.png")

# -------------------------
# Save Results
# -------------------------
save_results(results)
