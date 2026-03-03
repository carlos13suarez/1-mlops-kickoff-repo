"""
Educational Goal:
- Why this module exists in an MLOps system: Evaluation metrics must be consistent across experiments and aligned with business objectives
- Responsibility (separation of concerns): This module owns metric computation logic, making it easy to swap metrics without touching training code
- Pipeline contract (inputs and outputs): Accepts FITTED model and test data, returns a single float metric for monitoring and alerting

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import numpy as np
import pandas as pd
from sklearn import metrics


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, problem_type: str) -> float:
    """
    Inputs:
    - model: FITTED Pipeline from train.py
    - X: Features (DataFrame)
    - y: Target (Series)
    - problem_type: "regression" or "classification"
    Outputs:
    - metric: Single float (RMSE for regression, F1 for classification)
    Why this contract matters for reliable ML delivery:
    - Single metric output simplifies production monitoring and alerting thresholds
    - Explicit problem_type ensures correct metric is used when models are swapped
    - Returning float enables automated model comparison and rollback decisions
    """
    print(f"[evaluate] Evaluating {problem_type} model on {len(X)} samples")  # TODO: replace with logging later
    
    # 1. Generate predictions (these will be in log space because of our train.py logic)
    y_pred_log = model.predict(X)
    
    # 2. Inverse transform to get actual dollar prices
    y_pred = np.expm1(y_pred_log)
    
    # --------------------------------------------------------
    # START STUDENT CODE: Evaluation Metrics
    # --------------------------------------------------------
    # We compute the metrics from your notebook to show in the logs
    mae = metrics.mean_absolute_error(y, y_pred)
    r2 = metrics.r2_score(y, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
    
    print(f"  - R-squared: {r2:.4f}")
    print(f"  - Mean Absolute Error: ${mae:,.2f}")
    print(f"  - Root Mean Squared Error: ${rmse:,.2f}")

    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    # Return the primary metric for the pipeline's records
    return float(rmse)