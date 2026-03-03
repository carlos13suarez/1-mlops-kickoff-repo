"""
Educational Goal:
- Why this module exists in an MLOps system: Model training logic must be reproducible, version-controlled, and isolated from experimentation code
- Responsibility (separation of concerns): This module owns model instantiation and training, always using a Pipeline to guarantee preprocessing + model are serialized together
- Pipeline contract (inputs and outputs): Accepts training data and UNFITTED preprocessor, returns FITTED Pipeline ready for deployment

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn import metrics


def train_model(X: pd.DataFrame, y: pd.Series, preprocessor, problem_type: str):
    """
    Inputs:
    - X_train: Training features (DataFrame)
    - y_train: Training target (Series)
    - preprocessor: UNFITTED ColumnTransformer from features.py
    - problem_type: "regression" or "classification"
    Outputs:
    - pipeline: FITTED Pipeline containing preprocessor + model
    Why this contract matters for reliable ML delivery:
    - Pipeline ensures preprocessing and model are always applied together (prevents train/serve skew)
    - Serializing the full Pipeline guarantees production predictions match training transformations
    - Explicit problem_type parameter makes model selection auditable and testable
    """
    print(f"[train] Starting K-Fold CV (5 folds) for {problem_type}")  # TODO: replace with logging later
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    
    # 1. Setup K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    # We iterate through folds to validate stability before final fit
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Define the estimator (Model 5 used LinearRegression)
        estimator = LinearRegression()

        # Build the Pipeline for this fold
        fold_pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", estimator)
        ])

        # Fit on log-transformed target (as per notebook logic)
        fold_pipeline.fit(X_train_fold, np.log1p(y_train_fold))

        # Predict and inverse transform
        y_pred = np.expm1(fold_pipeline.predict(X_val_fold))
        
        # Track RMSE for this fold
        rmse = np.sqrt(metrics.mean_squared_error(y_val_fold, y_pred))
        rmse_scores.append(rmse)
        print(f"  - Fold {fold} RMSE: {rmse:.2f}")

    print(f"[train] Average CV RMSE: {np.mean(rmse_scores):.2f}")

    # 2. Final Fit
    # After CV, we train on the ENTIRE dataset to produce the final production model
    final_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LinearRegression())
    ])
    
    # We fit on the full data using the log transform
    final_pipeline.fit(X, np.log1p(y))

    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return final_pipeline