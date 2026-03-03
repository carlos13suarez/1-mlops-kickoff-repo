"""
Educational Goal:
- Why this module exists in an MLOps system: Feature engineering bugs cause data leakage (using test set statistics) which inflates validation metrics and causes production failures
- Responsibility (separation of concerns): This module builds UNFITTED transformation recipes (ColumnTransformer) that will be fitted only on training data inside the Pipeline
- Pipeline contract (inputs and outputs): Returns a ColumnTransformer recipe configured with student-specified transformations, preventing leakage by deferring .fit() to train.py

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from typing import Optional, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.preprocessing import StandardScaler


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
n_bins: int = 5 # Adjusted to 5 bins to match your Notebook's K-Fold strategy
):
    """
    Inputs:
    - quantile_bin_cols: Numeric columns to bin into quantiles (prevents overfitting to outliers)
    - categorical_onehot_cols: Categorical columns to one-hot encode
    - numeric_passthrough_cols: Numeric columns to pass through without transformation
    - n_bins: Number of quantile bins for KBinsDiscretizer
    Outputs:
    - ColumnTransformer: UNFITTED feature preprocessing recipe
    Why this contract matters for reliable ML delivery:
    - Returns UNFITTED transformer to guarantee train/test split happens before any .fit() calls
    - ColumnTransformer ensures consistent feature order and prevents missing columns in production
    - Explicit configuration prevents silent feature engineering bugs
    """
    print(f"[features] Building recipe: {len(quantile_bin_cols or [])} bin, {len(categorical_onehot_cols or [])} ohe, {len(numeric_passthrough_cols or [])} pass")  # TODO: replace with logging later
    
    transformers = []
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    
    # 1. Quantile Binning (e.g., for 'area')
    if quantile_bin_cols:
        # We use onehot-dense so it integrates cleanly with the other encoded features
        kbd = KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy="quantile")
        transformers.append(("quantile_bin", kbd, quantile_bin_cols))

    # 2. Categorical One-Hot Encoding (e.g., for 'furnishingstatus')
    if categorical_onehot_cols:
        # Note: 'sparse_output' is for newer sklearn; 'sparse' for older. 
        # The prompt requires a try/except for maximum compatibility.
        try:
            ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        except TypeError:
            ohe = OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")
        
        transformers.append(("cat_onehot", ohe, categorical_onehot_cols))

    # 3. Numeric Passthrough + Scaling (e.g., for 'bedrooms', 'bathrooms', etc.)
    if numeric_passthrough_cols:
        # Standardizing numeric inputs is best practice for Linear Regression
        scaler = StandardScaler()
        transformers.append(("num_scaler", scaler, numeric_passthrough_cols))
        
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    # Build ColumnTransformer (remainder="drop" ensures only specified columns are used)
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )
    
    print(f"[features] Feature recipe built with {len(transformers)} transformer groups")
    return preprocessor