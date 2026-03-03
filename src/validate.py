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

    # 1. Check required columns exist
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"[validate] VALIDATION FAILED: Missing required columns: {missing}")

    # 2. Check for Nulls in critical columns
    # We check the entire set of required columns for any NaN values
    null_counts = df[required_columns].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].index.tolist()
    
    if cols_with_nulls:
        raise ValueError(f"[validate] VALIDATION FAILED: The following columns contain null values: {cols_with_nulls}")

    # 3. Check data types for Numeric features
    # Since we are doing Regression, we expect the target and area to be numeric
    # This prevents the ColumnTransformer from failing later
    numeric_columns = [
        "price", "area", "bedrooms", 
        "bathrooms", "stories", "parking"
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"[validate] VALIDATION FAILED: Column '{col}' must be numeric (found {df[col].dtype}).")
        
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    print("[validate] Validation passed")
    return True