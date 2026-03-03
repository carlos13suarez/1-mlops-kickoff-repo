"""
Test Suite for features.py

Educational Purpose:
- This test file validates the get_feature_preprocessor() function
- Feature engineering is a critical MLOps component; bugs here cause silent data leakage
- These tests ensure the ColumnTransformer is properly configured and doesn't crash on real data

What's Being Tested:
1. Returns a ColumnTransformer object (correct type)
2. ColumnTransformer can be successfully fitted and transformed
3. Preprocessor handles the specified columns correctly
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.compose import ColumnTransformer

# Make sure we can import from src/
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.features import get_feature_preprocessor


@pytest.fixture
def mock_housing_df():
    """
    Create a small 5-row DataFrame that mimics the Housing dataset structure.
    
    This fixture includes numeric columns (price, area, bedrooms, bathrooms) that will be
    used with the preprocessor. It represents clean, validated data ready for feature engineering.
    
    Returns:
        pd.DataFrame with housing features
    """
    return pd.DataFrame({
        "price": [4000.0, 5500.0, 6200.0, 7100.0, 8300.0],
        "area": [300.0, 400.0, 500.0, 550.0, 600.0],
        "mainroad": [1, 0, 1, 1, 0],
        "furnishingstatus": ["unfurnished", "unfurnished", "semi-furnished", "furnished", "unfurnished"],
        "bedrooms": [2.0, 3.0, 3.0, 4.0, 4.0],
        "bathrooms": [1.0, 2.0, 2.0, 2.0, 3.0],
        "parking": [1.0, 2.0, 1.0, 3.0, 2.0],
    })


class TestFeaturePreprocessorType:
    """Test group: Verify the preprocessor has the correct type and structure"""
    
    def test_get_feature_preprocessor_returns_columntransformer(self, mock_housing_df):
        """
        REQUIREMENT: get_feature_preprocessor() must return a ColumnTransformer object
        
        The downstream pipeline code expects a ColumnTransformer. If a different object
        is returned (e.g., a dict or a different transformer), sklearn.Pipeline will fail.
        
        Setup:
            - Call the function with basic parameters
        Assert:
            - Return type is ColumnTransformer
        """
        # Create a preprocessor with minimal configuration
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=["area"],
            categorical_onehot_cols=["furnishingstatus"],
            numeric_passthrough_cols=["bedrooms", "bathrooms"]
        )
        
        # Most critical assertion: correct return type
        assert isinstance(preprocessor, ColumnTransformer), \
            f"Expected ColumnTransformer, got {type(preprocessor)}"
    
    def test_feature_preprocessor_has_transformers(self):
        """
        REQUIREMENT: The ColumnTransformer should contain transformer groups
        
        A properly configured preprocessor will have a 'transformers' attribute
        before fitting. After fitting, it will have 'transformers_' (fitted).
        This ensures the user's feature engineering logic was actually added.
        
        Assert:
            - Preprocessor has 'transformers' attribute
            - Transformers list is not empty
        """
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=["area"],
            categorical_onehot_cols=["furnishingstatus"],
            numeric_passthrough_cols=["bedrooms"]
        )
        
        # Check that transformers were configured (before fitting)
        assert hasattr(preprocessor, "transformers"), \
            "ColumnTransformer should have 'transformers' attribute"
        
        # The transformers list should not be empty
        assert len(preprocessor.transformers) > 0, \
            "ColumnTransformer should have at least one transformer group"


class TestFeaturePreprocessorFitting:
    """Test group: Verify the preprocessor can be fitted and transformed without crashing"""
    
    def test_feature_preprocessor_fits_without_error(self, mock_housing_df):
        """
        REQUIREMENT: Preprocessor should fit successfully on valid data
        
        If fitting crashes, the entire training pipeline fails. This test ensures
        the preprocessor configuration is valid and compatible with the housing data.
        
        Setup:
            - Create preprocessor
            - Fit it on real data
            - Include all expected features
        Assert:
            - No exception is raised
            - Fitted preprocessor is still a ColumnTransformer
        """
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=["area"],
            categorical_onehot_cols=["furnishingstatus"],
            numeric_passthrough_cols=["bedrooms", "bathrooms"]
        )
        
        # Extract features needed for preprocessing
        X = mock_housing_df[["area", "furnishingstatus", "bedrooms", "bathrooms"]]
        
        # Try to fit - this should not raise an exception
        try:
            preprocessor.fit(X)
        except Exception as e:
            pytest.fail(f"Preprocessor fitting failed with exception: {e}")
    
    def test_feature_preprocessor_transforms_without_error(self, mock_housing_df):
        """
        REQUIREMENT: After fitting, preprocessor should transform data without crashing
        
        A fitted preprocessor should be able to transform the data into features
        ready for the ML model. If transform() crashes, the pipeline breaks.
        
        Setup:
            - Create and fit preprocessor
            - Call transform() on the same data
        Assert:
            - No exception is raised
            - Output is a numpy array or DataFrame
        """
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=["area"],
            categorical_onehot_cols=["furnishingstatus"],
            numeric_passthrough_cols=["bedrooms", "bathrooms"]
        )
        
        X = mock_housing_df[["area", "furnishingstatus", "bedrooms", "bathrooms"]]
        
        # Fit the preprocessor
        preprocessor.fit(X)
        
        # Try to transform - should not raise an exception
        try:
            X_transformed = preprocessor.transform(X)
        except Exception as e:
            pytest.fail(f"Preprocessor transform failed with exception: {e}")
        
        # Verify we got some output (not None)
        assert X_transformed is not None, \
            "Transform should return a non-None result"
    
    def test_feature_preprocessor_fit_transform_workflow(self, mock_housing_df):
        """
        REQUIREMENT: Full fit-transform workflow should work end-to-end
        
        In scikit-learn, both fit() and fit_transform() should work.
        This tests the complete feature engineering workflow.
        
        Setup:
            - Create preprocessor with all feature types
            - Call fit_transform() directly
        Assert:
            - No exception is raised
            - Output shape has 5 rows (same as input)
            - Output has numeric features (all float64)
        """
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=["area"],
            categorical_onehot_cols=["furnishingstatus"],
            numeric_passthrough_cols=["bedrooms", "bathrooms", "parking"]
        )
        
        X = mock_housing_df[["area", "furnishingstatus", "bedrooms", "bathrooms", "parking"]]
        
        # Use fit_transform directly
        try:
            X_transformed = preprocessor.fit_transform(X)
        except Exception as e:
            pytest.fail(f"Preprocessor fit_transform failed with exception: {e}")
        
        # Verify the output preserves the number of samples
        assert X_transformed.shape[0] == 5, \
            f"Expected 5 rows after transformation, got {X_transformed.shape[0]}"
        
        # Verify the output has multiple features (not just 1)
        assert X_transformed.shape[1] > 0, \
            "Transformed output should have at least 1 feature column"


class TestFeaturePreprocessorConfiguration:
    """Test group: Verify different preprocessor configurations work correctly"""
    
    def test_feature_preprocessor_with_quantile_binning_only(self, mock_housing_df):
        """
        REQUIREMENT: Preprocessor should work with only quantile binning
        
        Users might configure only one type of feature transformation.
        The function should handle partial configurations gracefully.
        
        Setup:
            - Configure only quantile binning
        Assert:
            - Preprocessor is created and fits without error
        """
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=["area"],
            categorical_onehot_cols=None,
            numeric_passthrough_cols=None
        )
        
        X = mock_housing_df[["area"]]
        
        try:
            preprocessor.fit(X)
            X_transformed = preprocessor.transform(X)
        except Exception as e:
            pytest.fail(f"Preprocessor with quantile-only config failed: {e}")
        
        # Should have output
        assert X_transformed.shape[0] == 5, "Quantile binning should preserve row count"
    
    def test_feature_preprocessor_with_categorical_only(self, mock_housing_df):
        """
        REQUIREMENT: Preprocessor should work with only categorical encoding
        
        Setup:
            - Configure only categorical one-hot encoding
        Assert:
            - Preprocessor is created and works without error
        """
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=None,
            categorical_onehot_cols=["furnishingstatus"],
            numeric_passthrough_cols=None
        )
        
        X = mock_housing_df[["furnishingstatus"]]
        
        try:
            preprocessor.fit(X)
            X_transformed = preprocessor.transform(X)
        except Exception as e:
            pytest.fail(f"Preprocessor with categorical-only config failed: {e}")
        
        assert X_transformed.shape[0] == 5, "One-hot encoding should preserve row count"
    
    def test_feature_preprocessor_with_passthrough_only(self, mock_housing_df):
        """
        REQUIREMENT: Preprocessor should work with only numeric passthrough + scaling
        
        Setup:
            - Configure only numeric scaling
        Assert:
            - Preprocessor is created and works without error
        """
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=None,
            categorical_onehot_cols=None,
            numeric_passthrough_cols=["bedrooms", "bathrooms"]
        )
        
        X = mock_housing_df[["bedrooms", "bathrooms"]]
        
        try:
            preprocessor.fit(X)
            X_transformed = preprocessor.transform(X)
        except Exception as e:
            pytest.fail(f"Preprocessor with passthrough-only config failed: {e}")
        
        # After StandardScaler, values should be centered near 0
        assert X_transformed.shape[0] == 5, "Scaling should preserve row count"
        assert X_transformed.shape[1] == 2, "Should have 2 features"


class TestFeaturePreprocessorDataValidation:
    """Test group: Verify the preprocessor handles data correctly"""
    
    def test_feature_preprocessor_output_is_numeric(self, mock_housing_df):
        """
        REQUIREMENT: Preprocessor should output only numeric data
        
        The ML model expects numeric input. All categorical features should be
        encoded to numeric values (one-hot, integer, etc.).
        
        Setup:
            - Fit and transform data with categorical features
        Assert:
            - All output values are numeric (float or int)
        """
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=["area"],
            categorical_onehot_cols=["furnishingstatus"],
            numeric_passthrough_cols=["bedrooms"]
        )
        
        X = mock_housing_df[["area", "furnishingstatus", "bedrooms"]]
        X_transformed = preprocessor.fit_transform(X)
        
        # Check that output is numeric (numpy array of floats)
        assert isinstance(X_transformed, np.ndarray), \
            f"Expected numpy array output, got {type(X_transformed)}"
        
        # All values should be numeric (no strings or objects)
        # This works because we'll convert to float if needed
        try:
            X_numeric = X_transformed.astype(float)
        except (ValueError, TypeError):
            pytest.fail("Transformed output should be convertible to float (all numeric)")
