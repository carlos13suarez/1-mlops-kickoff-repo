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
    n_bins: int = 3
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
    print("[features] Building feature preprocessing recipe")  # TODO: replace with logging later
    
    # Handle None defaults
    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []
    
    transformers = []
    
    # Quantile binning for numeric features (reduces overfitting to outliers)
    if quantile_bin_cols:
        transformers.append((
            "quantile_bin",
            KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile'),
            quantile_bin_cols
        ))
    
    # One-hot encoding for categorical features
    if categorical_onehot_cols:
        # Backwards compatibility: sparse_output replaced sparse in scikit-learn 1.2+
        try:
            encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)
        
        transformers.append((
            "onehot",
            encoder,
            categorical_onehot_cols
        ))
    
    # Numeric passthrough (no transformation)
    if numeric_passthrough_cols:
        transformers.append((
            "numeric_passthrough",
            "passthrough",
            numeric_passthrough_cols
        ))
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add custom feature transformations from your notebook
    # Why: Real-world features often need domain-specific engineering (scaling, interactions, text vectorization)
    # Examples:
    # 1. Add StandardScaler for features that need normalization:
    #    transformers.append(("scaler", StandardScaler(), ['feature_1', 'feature_2']))
    # 2. Add polynomial features:
    #    from sklearn.preprocessing import PolynomialFeatures
    #    transformers.append(("poly", PolynomialFeatures(degree=2), ['feature_1']))
    # 3. Add text vectorization:
    #    from sklearn.feature_extraction.text import TfidfVectorizer
    #    transformers.append(("tfidf", TfidfVectorizer(max_features=100), 'text_column'))
    # 4. Add custom transformers:
    #    from sklearn.preprocessing import FunctionTransformer
    #    transformers.append(("log_transform", FunctionTransformer(np.log1p), ['skewed_feature']))
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
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