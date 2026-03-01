"""
Educational Goal:
- Why this module exists in an MLOps system: Data quality issues (missing values, outliers, inconsistencies) cause 80% of production ML failures
- Responsibility (separation of concerns): This module owns data cleaning transformations that must happen BEFORE feature engineering
- Pipeline contract (inputs and outputs): Accepts raw DataFrame, returns cleaned DataFrame ready for validation and feature extraction

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Inputs:
    - df_raw: Raw DataFrame from load_data module
    - target_column: Name of the target variable (needed for target-specific cleaning)
    Outputs:
    - df_clean: Cleaned DataFrame (imputed, filtered, deduplicated)
    Why this contract matters for reliable ML delivery:
    - Explicit cleaning stage allows auditing data quality before model training
    - Separating cleaning from feature engineering prevents leakage (e.g., imputing with test set statistics)
    """
    print(f"[clean_data] Cleaning {len(df_raw)} rows")  # TODO: replace with logging later
    
    # Baseline: Safe copy (identity transformation)
    df_clean = df_raw.copy()
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook's data cleaning logic here
    # Why: Every dataset has unique quality issues that require domain-specific cleaning
    # Examples:
    # 1. df_clean = df_clean.dropna(subset=[target_column])
    # 2. df_clean = df_clean[df_clean['age'] > 0]
    # 3. df_clean['price'] = df_clean['price'].fillna(df_clean['price'].median())
    # 4. df_clean = df_clean.drop_duplicates()
    # 5. df_clean = df_clean[df_clean['date'] >= '2020-01-01']
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    print(f"[clean_data] Cleaning complete: {len(df_clean)} rows remaining")
    return df_clean