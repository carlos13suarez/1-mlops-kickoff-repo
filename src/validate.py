"""
Educational Goal:
- Why this module exists in an MLOps system: Fail-fast validation prevents silent errors that corrupt downstream model training and predictions
- Responsibility (separation of concerns): This module enforces schema contracts and data quality invariants
- Pipeline contract (inputs and outputs): Accepts DataFrame and required columns list, raises exceptions on critical failures, returns True if valid

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df: DataFrame to validate
    - required_columns: List of column names that MUST exist
    Outputs:
    - True if valid (raises exception otherwise)
    Why this contract matters for reliable ML delivery:
    - Early validation prevents cascading failures that are expensive to debug
    - Explicit schema checks catch upstream data provider changes immediately
    """
    print(f"[validate] Validating DataFrame with {len(df)} rows")  # TODO: replace with logging later
    
    # Fail-fast: Empty DataFrame
    if df.empty:
        raise ValueError("[validate] CRITICAL: DataFrame is empty! Cannot proceed with model training.")
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add dataset-specific validation checks
    # Why: Production pipelines must enforce business rules and data contracts
    # Examples:
    # 1. Check required columns exist:
    #    missing = set(required_columns) - set(df.columns)
    #    if missing:
    #        raise ValueError(f"Missing columns: {missing}")
    # 2. Check value ranges:
    #    if (df['age'] < 0).any():
    #        raise ValueError("Age cannot be negative")
    # 3. Check data types:
    #    if df['price'].dtype not in ['int64', 'float64']:
    #        raise ValueError("Price must be numeric")
    # 4. Check for excessive missingness:
    #    if df.isnull().sum().sum() / df.size > 0.5:
    #        raise ValueError("More than 50% missing values")
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    print("[validate] Validation passed")
    return True