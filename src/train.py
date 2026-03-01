"""
Educational Goal:
- Why this module exists in an MLOps system: Model training logic must be reproducible, version-controlled, and isolated from experimentation code
- Responsibility (separation of concerns): This module owns model instantiation and training, always using a Pipeline to guarantee preprocessing + model are serialized together
- Pipeline contract (inputs and outputs): Accepts training data and UNFITTED preprocessor, returns FITTED Pipeline ready for deployment

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression


def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor, problem_type: str):
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
    print(f"[train] Training {problem_type} model on {len(X_train)} samples")  # TODO: replace with logging later
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Replace baseline model with your notebook's tuned model
    # Why: Model selection and hyperparameters vary by dataset and business requirements
    # Examples:
    # 1. Regression with hyperparameters:
    #    from sklearn.ensemble import RandomForestRegressor
    #    estimator = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    # 2. Classification with hyperparameters:
    #    from sklearn.ensemble import GradientBoostingClassifier
    #    estimator = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    # 3. Add hyperparameter tuning:
    #    from sklearn.model_selection import GridSearchCV
    #    estimator = GridSearchCV(Ridge(), param_grid={'alpha': [0.1, 1.0, 10.0]}, cv=5)
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    # Baseline model selection
    if problem_type == "regression":
        estimator = Ridge()
    elif problem_type == "classification":
        estimator = LogisticRegression(max_iter=500)
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")
    
    # Build Pipeline: preprocessor + model
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", estimator)
    ])
    
    # Fit pipeline (preprocessor.fit_transform + model.fit happen internally)
    pipeline.fit(X_train, y_train)
    
    print(f"[train] Training complete")
    return pipeline