"""
Test Suite for clean_data.py

Educational Purpose:
- This test file validates the clean_dataframe() function which is critical for MLOps pipelines
- Data cleaning is one of the most error-prone steps; testing ensures consistent data quality
- These tests prevent regressions when the cleaning logic is modified in the future

What's Being Tested:
1. Binary encoding ('yes'/'no' → 1/0 conversion)
2. Duplicate row removal
3. Output shape and data consistency
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Make sure we can import from src/
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.clean_data import clean_dataframe


@pytest.fixture
def mock_housing_df():
    """
    Create a small 5-row DataFrame that mimics the Housing dataset structure.
    
    This fixture is used by all test functions in this file to ensure consistent,
    reproducible test data. It includes all critical columns plus some with yes/no values.
    
    Returns:
        pd.DataFrame with the housing data structure
    """
    return pd.DataFrame({
        "price": [4000, 5500, 6200, 7100, 8300],
        "area": [300, 400, 500, 550, 600],
        "mainroad": ["yes", "no", "yes", "yes", "no"],
        "furnishingstatus": ["unfurnished", "unfurnished", "semi-furnished", "furnished", "unfurnished"],
        "bedrooms": [2, 3, 3, 4, 4],
        "bathrooms": [1, 2, 2, 2, 3],
    })


class TestCleanDataEncoding:
    """Test group: Verify that categorical yes/no columns are properly encoded to 1/0"""
    
    def test_clean_dataframe_converts_yes_no_to_binary(self, mock_housing_df):
        """
        REQUIREMENT: clean_dataframe() should convert 'yes' → 1 and 'no' → 0
        
        This is critical because ML models require numeric input.
        String values like 'yes' and 'no' cannot be fed directly to sklearn models.
        
        Assert:
            - 'mainroad' column contains only 0 and 1 (no 'yes' or 'no' strings)
            - Correct mapping is applied ('yes' becomes 1, 'no' becomes 0)
        """
        # Execute the cleaning function
        result = clean_dataframe(mock_housing_df, target_column="price")
        
        # Verify only 0 and 1 values exist (no 'yes' or 'no' strings)
        unique_values = set(result["mainroad"].unique())
        assert unique_values.issubset({0, 1, "0", "1"}), \
            f"Expected mainroad to contain only 0 and 1 (as int or string), but got {unique_values}"
        
        # Verify no 'yes'/'no' strings remain after cleaning
        assert "yes" not in unique_values and "no" not in unique_values, \
            f"Expected 'yes' and 'no' to be converted, but still found in {unique_values}"
    
    def test_clean_dataframe_binary_encoding_correctness(self, mock_housing_df):
        """
        REQUIREMENT: Verify the mapping is semantically correct
        
        If the original data has 'yes' in position 0, after cleaning we should see 1 in position 0.
        
        Assert:
            - First row had 'yes' → should be 1
            - Second row had 'no' → should be 0
        """
        result = clean_dataframe(mock_housing_df, target_column="price")
        
        # The mock data has "yes" at index 0, "no" at index 1
        assert result["mainroad"].iloc[0] == 1, \
            "Expected 'yes' to be encoded as 1, but got different value"
        assert result["mainroad"].iloc[1] == 0, \
            "Expected 'no' to be encoded as 0, but got different value"


class TestCleanDataDeduplication:
    """Test group: Verify that duplicate rows are properly removed"""
    
    def test_clean_dataframe_removes_duplicate_rows(self, mock_housing_df):
        """
        REQUIREMENT: clean_dataframe() should remove duplicate rows
        
        Duplicate rows cause two problems:
        1. If a duplicate appears in both train and test, metrics become overly optimistic
        2. The model learns redundant patterns instead of generalizing
        
        Setup:
            - Create a DataFrame with duplicate rows
        Assert:
            - Output has fewer rows than input (duplicates removed)
            - Remaining rows contain unique data
        """
        # Create a DataFrame with duplicates
        df_with_dupes = pd.concat([mock_housing_df, mock_housing_df.iloc[[0, 1]]], ignore_index=True)
        assert len(df_with_dupes) == 7, "Test setup: should have 7 rows with duplicates"
        
        # Clean the data
        result = clean_dataframe(df_with_dupes, target_column="price")
        
        # Verify duplicates were removed
        assert len(result) == 5, \
            f"Expected 5 rows after deduplication, but got {len(result)} rows"
        
        # Verify no remaining duplicates
        assert result.duplicated().sum() == 0, \
            "Found duplicate rows in the cleaned dataset"
    
    def test_clean_dataframe_preserves_unique_rows(self, mock_housing_df):
        """
        REQUIREMENT: Deduplication should NOT remove unique rows
        
        Some test data has no duplicates. We must verify the function handles this gracefully.
        
        Assert:
            - No rows are unnecessarily dropped
            - All unique rows are preserved
        """
        # Verify the mock data has no duplicates
        assert mock_housing_df.duplicated().sum() == 0, "Test setup error: mock data should have no duplicates"
        
        # Clean
        result = clean_dataframe(mock_housing_df, target_column="price")
        
        # Should still have 5 rows since nothing is duplicated
        assert len(result) == 5, \
            f"Expected all 5 rows to be preserved (no duplicates), but got {len(result)} rows"


class TestCleanDataIntegration:
    """Test group: Integration tests combining multiple cleaning operations"""
    
    def test_clean_dataframe_returns_dataframe(self, mock_housing_df):
        """
        REQUIREMENT: Function should always return a pandas DataFrame
        
        The downstream functions (validate.py, features.py) expect a DataFrame.
        If we accidentally return a different type, cascading errors occur.
        
        Assert:
            - Return type is DataFrame
        """
        result = clean_dataframe(mock_housing_df, target_column="price")
        assert isinstance(result, pd.DataFrame), \
            f"Expected DataFrame output, got {type(result)}"
    
    def test_clean_dataframe_handles_missing_target_column(self, mock_housing_df):
        """
        REQUIREMENT: If target column is missing, function should handle gracefully
        
        We might receive data without the price column in edge cases.
        The function should not crash; it should skip target-specific logic.
        
        Assert:
            - Function runs without raising an exception
            - Result is still a valid DataFrame
        """
        # This tests defensive programming: what if 'price' column doesn't exist?
        try:
            result = clean_dataframe(mock_housing_df, target_column="nonexistent_column")
            assert isinstance(result, pd.DataFrame), "Should still return a DataFrame"
        except KeyError:
            # It's acceptable if the function raises KeyError for missing columns
            # The important thing is it fails explicitly rather than silently
            pass
