import json
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from data_loader import load_and_split_data

def evaluate_best_model(study, X_train, X_test, y_train, y_test, optimization_time):
    """
    Evaluate the best model from the Optuna study and generate results summary.

    Args:
        study: Optuna study object
        X_train, X_test, y_train, y_test: Train/test data splits
        optimization_time: Time taken for optimization in seconds

    Returns:
        dict: Results summary dictionary
    """
    # Get best parameters
    best_params = study.best_params
    best_params.update({
        "objective": "reg:squarederror",
        "random_state": 42
    })

    # Train best model
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_pred)

    # Compile results
    results = {
        "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_trials_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "best_cv_rmse": np.sqrt(-study.best_value),
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "best_params": best_params,
        "optimization_time_seconds": optimization_time
    }

    return results, model

def save_results(results, output_path="../outputs/results.json"):
    """
    Save results dictionary to JSON file.

    Args:
        results: Results dictionary
        output_path: Path to save the JSON file
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

def generate_baseline_comparison(X_train, X_test, y_train, y_test):
    """
    Generate baseline model performance for comparison.

    Args:
        X_train, X_test, y_train, y_test: Train/test data splits

    Returns:
        dict: Baseline performance metrics
    """
    # Baseline XGBoost with default parameters
    baseline_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    )

    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict(X_test)

    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
    baseline_r2 = r2_score(y_test, baseline_preds)

    return {
        "baseline_rmse": baseline_rmse,
        "baseline_r2": baseline_r2
    }