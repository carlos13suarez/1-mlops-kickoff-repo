"""
Test Suite for validate.py

Educational Purpose:
- This test file validates the validate_dataframe() function which enforces data quality gates
- Validation is a "fail-fast" mechanism that prevents bad data from entering the ML pipeline
- These tests ensure the function correctly rejects invalid data while accepting valid data

What's Being Tested:
1. Valid data passes validation and returns True
2. Missing required columns raises ValueError
3. Null values in required columns are detected
4. Data type validation works correctly
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Make sure we can import from src/
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.validate import validate_dataframe


@pytest.fixture
def mock_housing_df():
    """
    Create a small 5-row DataFrame that mimics the Housing dataset structure.
    
    This fixture represents VALID, clean housing data with all required columns present,
    no missing values, and proper data types. It's used as the baseline for tests.
    
    Returns:
        pd.DataFrame with valid housing data
    """
    return pd.DataFrame({
        "price": [4000.0, 5500.0, 6200.0, 7100.0, 8300.0],
        "area": [300.0, 400.0, 500.0, 550.0, 600.0],
        "mainroad": [1, 0, 1, 1, 0],  # Already binary encoded (0/1)
        "furnishingstatus": ["unfurnished", "unfurnished", "semi-furnished", "furnished", "unfurnished"],
        "bedrooms": [2, 3, 3, 4, 4],
        "bathrooms": [1, 2, 2, 2, 3],
    })


class TestValidateDataframeValidInput:
    """Test group: Verify that valid data is accepted"""
    
    def test_valid_dataframe_returns_true(self, mock_housing_df):
        """
        REQUIREMENT: validate_dataframe() should return True for valid data
        
        Valid data is defined as:
        - Contains all required columns
        - No null values in required columns
        - Required columns have correct data types (numeric for numeric_columns)
        
        This test ensures the validation function is not overly strict.
        
        Assert:
            - Function returns True
            - No exception is raised
        """
        required_columns = ["price", "area", "mainroad", "bedrooms", "bathrooms"]
        
        # This should not raise an exception
        result = validate_dataframe(mock_housing_df, required_columns)
        
        # Should return True when data is valid
        assert result is True, \
            f"Expected validate_dataframe to return True for valid data, got {result}"
    
    def test_valid_dataframe_is_not_empty(self, mock_housing_df):
        """
        REQUIREMENT: We should test on non-empty data
        
        An empty DataFrame should fail validation earlier (checked in the validate_dataframe code).
        This test ensures our mock data is suitable for validation testing.
        
        Assert:
            - Mock DataFrame is not empty
        """
        assert not mock_housing_df.empty, "Mock data should not be empty"
        assert len(mock_housing_df) == 5, "Mock data should have exactly 5 rows"


class TestValidateDataframeInvalidInput:
    """Test group: Verify that invalid data is properly rejected"""
    
    def test_missing_required_column_raises_error(self, mock_housing_df):
        """
        REQUIREMENT: If a required column is missing, raise ValueError
        
        This is a critical validation. If the dataset is missing the 'price' column,
        we cannot train a regression model. The validation function must catch this early.
        
        Setup:
            - Define ['price', 'area'] as required
            - Provide DataFrame that's missing 'price'
        Assert:
            - ValueError is raised
            - Error message mentions missing columns
        """
        # Remove the 'price' column from the data
        df_missing_price = mock_housing_df.drop(columns=["price"])
        
        required_columns = ["price", "area", "mainroad"]
        
        # Should raise ValueError when 'price' is missing
        with pytest.raises(ValueError) as exc_info:
            validate_dataframe(df_missing_price, required_columns)
        
        # Verify the error message is informative
        assert "price" in str(exc_info.value).lower(), \
            f"Error message should mention the missing column 'price': {exc_info.value}"
        assert "missing" in str(exc_info.value).lower(), \
            f"Error message should mention that columns are missing: {exc_info.value}"
    
    def test_multiple_missing_columns_are_detected(self, mock_housing_df):
        """
        REQUIREMENT: If multiple columns are missing, all should be reported
        
        If we're missing both 'price' and 'area', the error message should mention both.
        This helps users quickly identify all data issues at once (not one-by-one).
        
        Setup:
            - Drop both 'price' and 'area' columns
            - Require both columns
        Assert:
            - ValueError is raised
            - Error message mentions both missing columns
        """
        df_missing_cols = mock_housing_df.drop(columns=["price", "area"])
        
        required_columns = ["price", "area", "mainroad"]
        
        with pytest.raises(ValueError) as exc_info:
            validate_dataframe(df_missing_cols, required_columns)
        
        error_msg = str(exc_info.value).lower()
        assert "price" in error_msg and "area" in error_msg, \
            f"Error should mention both missing columns: {exc_info.value}"
    
    def test_null_values_in_required_columns_raises_error(self, mock_housing_df):
        """
        REQUIREMENT: If required columns contain null values, raise ValueError
        
        ML models cannot process NaN/Null values. Having nulls in required columns
        means the data is incomplete and cannot be used for training.
        
        Setup:
            - Introduce a null value in the 'price' column
            - Attempt validation
        Assert:
            - ValueError is raised
            - Error message indicates null values
        """
        # Introduce a null value
        df_with_null = mock_housing_df.copy()
        df_with_null.loc[0, "price"] = None
        
        required_columns = ["price", "area", "mainroad"]
        
        with pytest.raises(ValueError) as exc_info:
            validate_dataframe(df_with_null, required_columns)
        
        error_msg = str(exc_info.value).lower()
        assert "null" in error_msg or "nan" in error_msg or "na" in error_msg, \
            f"Error should mention null/NaN values: {exc_info.value}"
    
    def test_numeric_column_with_wrong_dtype_raises_error(self, mock_housing_df):
        """
        REQUIREMENT: If a numeric column has the wrong dtype, raise ValueError
        
        The validate function checks that numeric columns (like 'price') are actually numeric.
        If someone passes 'price' as strings, this should be caught immediately.
        
        Setup:
            - Convert 'price' column to string type
            - Attempt validation (price should be numeric)
        Assert:
            - ValueError is raised
            - Error message indicates dtype issue
        """
        df_wrong_dtype = mock_housing_df.copy()
        # Convert price to string (simulating bad upstream data)
        df_wrong_dtype["price"] = df_wrong_dtype["price"].astype(str)
        
        required_columns = ["price", "area", "mainroad"]
        
        with pytest.raises(ValueError) as exc_info:
            validate_dataframe(df_wrong_dtype, required_columns)
        
        error_msg = str(exc_info.value).lower()
        # The error should mention dtype or numeric issues
        assert "numeric" in error_msg or "dtype" in error_msg or "type" in error_msg, \
            f"Error should mention data type issues: {exc_info.value}"


class TestValidateDataframeEdgeCases:
    """Test group: Edge cases and boundary conditions"""
    
    def test_empty_required_columns_list(self, mock_housing_df):
        """
        REQUIREMENT: Validate function should handle empty required columns list
        
        While unusual, if someone passes an empty list, the function should not crash.
        
        Assert:
            - Function executes without raising an exception
            - Returns True (no required columns means nothing to validate)
        """
        # Empty required columns list
        result = validate_dataframe(mock_housing_df, [])
        
        # Empty requirements should pass (nothing to validate against)
        assert result is True, \
            "Empty required columns list should pass validation"
    
    def test_required_columns_with_extra_columns_in_data(self, mock_housing_df):
        """
        REQUIREMENT: Data can have extra columns beyond required ones
        
        The function should only validate that required columns exist.
        Extra columns (not in the required list) should be ignored and allowed.
        
        Assert:
            - Validation passes even with extra columns present
        """
        required_columns = ["price", "area"]
        
        # The mock data has more columns than required (bedrooms, bathrooms, furnishingstatus)
        # This should be fine
        result = validate_dataframe(mock_housing_df, required_columns)
        
        assert result is True, \
            "Validation should pass when data has extra columns beyond required ones"
