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
from sklearn.metrics import mean_squared_error, f1_score


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> float:
    """
    Inputs:
    - model: FITTED Pipeline from train.py
    - X_test: Test features (DataFrame)
    - y_test: Test target (Series)
    - problem_type: "regression" or "classification"
    Outputs:
    - metric: Single float (RMSE for regression, F1 for classification)
    Why this contract matters for reliable ML delivery:
    - Single metric output simplifies production monitoring and alerting thresholds
    - Explicit problem_type ensures correct metric is used when models are swapped
    - Returning float enables automated model comparison and rollback decisions
    """
    print(f"[evaluate] Evaluating {problem_type} model on {len(X_test)} samples")  # TODO: replace with logging later
    
    # Generate predictions using the full Pipeline (preprocess + predict)
    y_pred = model.predict(X_test)
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Customize evaluation metrics to match your business objective
    # Why: Different problems require different metrics (precision vs recall, MAE vs RMSE, custom business metrics)
    # Examples:
    # 1. Regression alternatives:
    #    from sklearn.metrics import mean_absolute_error, r2_score
    #    metric = mean_absolute_error(y_test, y_pred)
    # 2. Classification alternatives:
    #    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    #    metric = roc_auc_score(y_test, y_pred)
    # 3. Multi-metric evaluation:
    #    metrics = {
    #        'rmse': mean_squared_error(y_test, y_pred, squared=False),
    #        'mae': mean_absolute_error(y_test, y_pred),
    #        'r2': r2_score(y_test, y_pred)
    #    }
    #    print(f"Metrics: {metrics}")
    # 4. Custom business metric:
    #    def business_cost(y_true, y_pred):
    #        # False positives cost $100, false negatives cost $500
    #        return custom_logic(y_true, y_pred)
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    # Baseline metric computation
    if problem_type == "regression":
        metric = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE
        print(f"[evaluate] RMSE: {metric:.4f}")
    elif problem_type == "classification":
        metric = f1_score(y_test, y_pred, average='weighted')  # Weighted F1 (handles multiclass)
        print(f"[evaluate] F1 Score (weighted): {metric:.4f}")
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")
    
    return metric