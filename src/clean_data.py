"""
Educational Goal:
- Why this module exists in an MLOps system: Data quality issues (missing values, outliers, inconsistencies) cause 80% of production ML failures
- Responsibility (separation of concerns): This module owns data cleaning transformations that must happen BEFORE feature engineering
- Pipeline contract (inputs and outputs): Accepts raw DataFrame, returns cleaned DataFrame ready for validation and feature extraction

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd
import numpy as np


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Inputs:
    - df_raw: Raw DataFrame from load_data module
    - target_column: price
    Outputs:
    - df_clean: Cleaned DataFrame (imputed, filtered, deduplicated)
    Why this contract matters for reliable ML delivery:
    - Explicit cleaning stage allows auditing data quality before model training
    - Separating cleaning from feature engineering prevents leakage (e.g., imputing with test set statistics)
    """
    print(f"[clean_data] Cleaning {len(df_raw)} rows")  # TODO: replace with logging later
    
    # 1. Defensive Copy
    # Always work on a copy to avoid modifying the original 'raw' dataframe in memory
    df_clean = df_raw.copy()
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    
    # 2. Deduplication (Uniqueness)
    # Duplicate rows can lead to over-optimistic model performance if they appear in both train/test
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    if len(df_clean) < initial_count:
        print(f"[clean_data] Dropped {initial_count - len(df_clean)} duplicate rows")

    # 3. Handling Missing Values (Completeness)
    # Most ML models (like Linear Regression) cannot handle NaN/Null values
    # We drop rows where the 'target' is missing because we can't learn from them
    if target_column in df_clean.columns:
        df_clean = df_clean.dropna(subset=[target_column])
    
    # For other features, we drop rows that are completely empty
    df_clean = df_clean.dropna(how='all')

    # 4. Binary Encoding (Format Standardization)
    # Converting human-readable strings to machine-readable integers
    binary_cols = [
        "mainroad", "guestroom", "basement", 
        "hotwaterheating", "airconditioning", "prefarea"
    ]
    existing_binary_cols = [col for col in binary_cols if col in df_clean.columns]
    if existing_binary_cols:
        print(f"[clean_data] Standardizing binary columns: {existing_binary_cols}")
        df_clean[existing_binary_cols] = df_clean[existing_binary_cols].replace({"yes": 1, "no": 0})

    # 5. Type Casting
    # Ensuring numeric columns are actually floats/ints (sometimes Kaggle loads them as objects)
    numeric_cols = ["area", "bedrooms", "bathrooms", "stories", "parking", "price"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 6. Validity Filtering (Domain Logic)
    # Filtering out data that is physically impossible (e.g., negative prices or 0 area)
    if "price" in df_clean.columns:
        df_clean = df_clean[df_clean["price"] > 0]
    if "area" in df_clean.columns:
        df_clean = df_clean[df_clean["area"] > 0]

    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    print(f"[clean_data] Cleaning complete: {len(df_clean)} rows remaining")
    return df_clean