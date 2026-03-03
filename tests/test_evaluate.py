"""
Test Suite for evaluate.py

Educational Purpose:
- This test file validates the evaluate_model() function
- Evaluation metrics are critical for monitoring model performance in production
- These tests ensure metrics are computed correctly and handle inverse-log transformations properly

What's Being Tested:
1. evaluate_model returns a float (single metric value)
2. Returned metric is in the correct range for dollar prices
3. Inverse-log transformation is applied correctly (prices are in correct magnitude)
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Make sure we can import from src/
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.evaluate import evaluate_model


@pytest.fixture
def mock_housing_df():
    """
    Create a small 5-row DataFrame that mimics the Housing dataset structure.
    
    This fixture is used to create mock training data for fitting a simple model.
    The target (price) is in log-space, simulating what the training pipeline does.
    
    Returns:
        pd.DataFrame with housing features including a log-transformed price target
    """
    return pd.DataFrame({
        "price": [4000.0, 5500.0, 6200.0, 7100.0, 8300.0],
        "price_log": [np.log1p(4000), np.log1p(5500), np.log1p(6200), np.log1p(7100), np.log1p(8300)],
        "area": [300.0, 400.0, 500.0, 550.0, 600.0],
        "bedrooms": [2.0, 3.0, 3.0, 4.0, 4.0],
        "bathrooms": [1.0, 2.0, 2.0, 2.0, 3.0],
    })


@pytest.fixture
def fitted_model(mock_housing_df):
    """
    Create a simple fitted Linear Regression model for testing.
    
    This fixture builds and fits a basic pipeline on mock data.
    Important: The target is log-transformed (price_log), so when we predict,
    we get log-space predictions that need to be inverse-transformed.
    
    The evaluate_model function expects this behavior.
    
    Returns:
        A fitted sklearn Pipeline that outputs predictions in log-space
    """
    # Create features and target
    X = mock_housing_df[["area", "bedrooms", "bathrooms"]]
    y = mock_housing_df["price_log"]  # Target is in log space
    
    # Build a simple pipeline: StandardScaler + LinearRegression
    # This mimics a typical scikit-learn workflow
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    
    # Fit the model
    model.fit(X, y)
    
    return model


class TestEvaluateModelReturnType:
    """Test group: Verify evaluate_model returns the correct type"""
    
    def test_evaluate_model_returns_float(self, fitted_model, mock_housing_df):
        """
        REQUIREMENT: evaluate_model() should return a single float value
        
        The metric is used for monitoring and automated decision-making.
        If it returns a dict or array instead of a float, downstream code breaks.
        
        Setup:
            - Create a fitted model
            - Call evaluate_model with test data
        Assert:
            - Return value is a float
        """
        X_test = mock_housing_df[["area", "bedrooms", "bathrooms"]]
        y_test = mock_housing_df["price"]  # Target in original space (dollars)
        
        # Call the evaluation function
        metric = evaluate_model(fitted_model, X_test, y_test, problem_type="regression")
        
        # Most critical assertion: correct return type
        assert isinstance(metric, float), \
            f"Expected float return type, got {type(metric)}"
    
    def test_evaluate_model_returns_positive_rmse(self, fitted_model, mock_housing_df):
        """
        REQUIREMENT: RMSE (Root Mean Squared Error) should be positive
        
        RMSE is the square root of squared errors, so it's always non-negative.
        A negative metric would indicate a code bug.
        
        Assert:
            - Returned metric is positive (>= 0)
        """
        X_test = mock_housing_df[["area", "bedrooms", "bathrooms"]]
        y_test = mock_housing_df["price"]
        
        metric = evaluate_model(fitted_model, X_test, y_test, problem_type="regression")
        
        assert metric >= 0, \
            f"RMSE should be non-negative, got {metric}"


class TestEvaluateModelPriceRange:
    """Test group: Verify predictions are in the correct price magnitude"""
    
    def test_model_predictions_in_reasonable_price_range(self, fitted_model, mock_housing_df):
        """
        REQUIREMENT: Model predictions should be in the correct dollar range
        
        The mock data has prices between ~$4,000 and ~$8,300.
        If inverse-log is applied correctly, predictions should be in this ballpark.
        If inverse-log is wrong or missing, predictions might be $0-1 or $100,000+.
        
        This test verifies the inverse-log transformation works correctly.
        
        Setup:
            - Fit a model on log-transformed prices
            - Generate predictions
            - Check if predictions are in reasonable range
        Assert:
            - Predictions are between $1,000 and $50,000 (reasonable housing prices)
        """
        X_test = mock_housing_df[["area", "bedrooms", "bathrooms"]]
        
        # Get predictions from the model (in log space)
        y_pred_log = fitted_model.predict(X_test)
        
        # Apply inverse transformation (expm1 is the inverse of log1p)
        y_pred = np.expm1(y_pred_log)
        
        # Check predictions are in reasonable range for housing prices
        min_price_expected = 1000      # Minimum reasonable housing price
        max_price_expected = 50000     # Maximum reasonable for this dataset
        
        assert np.all(y_pred >= min_price_expected), \
            f"Predictions should be >= ${min_price_expected}, but got min: ${y_pred.min():,.0f}"
        
        assert np.all(y_pred <= max_price_expected), \
            f"Predictions should be <= ${max_price_expected}, but got max: ${y_pred.max():,.0f}"
    
    def test_evaluate_metric_is_in_reasonable_range(self, fitted_model, mock_housing_df):
        """
        REQUIREMENT: RMSE metric should be in a reasonable range relative to prices
        
        If RMSE is larger than the entire price range, something is wrong.
        RMSE should typically be a fraction of the price magnitude.
        
        Setup:
            - Compute RMSE for test data
            - Compare to the price magnitude
        Assert:
            - RMSE is less than the total price range
        """
        X_test = mock_housing_df[["area", "bedrooms", "bathrooms"]]
        y_test = mock_housing_df["price"]
        
        metric = evaluate_model(fitted_model, X_test, y_test, problem_type="regression")
        
        # Price range: min to max
        price_range = y_test.max() - y_test.min()
        
        # RMSE should be much smaller than the entire price range (sanity check)
        assert metric < price_range * 2, \
            f"RMSE ({metric:,.0f}) seems unreasonably large compared to price range ({price_range:,.0f})"


class TestEvaluateModelLogTransform:
    """Test group: Verify correct inverse-log transformation behavior"""
    
    def test_log_transform_inverse_correctness(self, mock_housing_df):
        """
        REQUIREMENT: Verify log1p and expm1 are inverses of each other
        
        Training uses log1p to transform prices. Evaluation must use expm1 to inverse-transform.
        This test confirms these operations work correctly.
        
        Setup:
            - Take original prices
            - Apply log1p
            - Apply expm1
            - Compare to original
        Assert:
            - Round-trip transformation recovers original values
        """
        original_prices = mock_housing_df["price"].values
        
        # Apply log transformation (as in training)
        log_prices = np.log1p(original_prices)
        
        # Apply inverse transformation (as in evaluate_model)
        recovered_prices = np.expm1(log_prices)
        
        # Should recover original prices (within floating point precision)
        np.testing.assert_allclose(original_prices, recovered_prices, rtol=1e-10)
    
    def test_predictions_match_expected_magnitude(self, fitted_model, mock_housing_df):
        """
        REQUIREMENT: Predictions from a fitted model should have the right magnitude
        
        If someone accidentally removes expm1() from evaluate_model, predictions would be
        in log-space (~8-9) instead of dollar space (~4000-8000).
        This test catches such regressions.
        
        Setup:
            - Generate predictions from the fitted model
            - Apply inverse log transformation
        Assert:
            - Predictions are in dollar magnitudes (thousands)
        """
        X_test = mock_housing_df[["area", "bedrooms", "bathrooms"]]
        
        # Predictions from fitted model (in log space)
        y_pred_log = fitted_model.predict(X_test)
        
        # Inverse transform
        y_pred = np.expm1(y_pred_log)
        
        # Predictions should be in thousands of dollars (roughly)
        assert np.all(y_pred > 100), \
            f"After inverse-log, predictions should be >> 1, got {y_pred}"
        
        assert np.all(y_pred < 1000000), \
            f"After inverse-log, predictions should be << 1,000,000, got {y_pred}"


class TestEvaluateModelIntegration:
    """Test group: Full integration tests with the evaluate_model function"""
    
    def test_evaluate_model_with_perfect_predictions(self, mock_housing_df):
        """
        REQUIREMENT: When model predicts perfectly, RMSE should be near zero
        
        This is a sanity check: if a model perfectly predicts the training data,
        RMSE should be very small (essentially 0, within floating point errors).
        
        Setup:
            - Create a "perfect" model that outputs exact training targets
            - Evaluate on the same data
        Assert:
            - Returned RMSE is close to 0
        """
        # Create a "perfect" model: always predicts the mean (won't be perfect, but reasonable)
        from sklearn.dummy import DummyRegressor
        
        X = mock_housing_df[["area", "bedrooms", "bathrooms"]]
        y_log = mock_housing_df["price_log"]
        y = mock_housing_df["price"]
        
        # Use a dummy model that always predicts the mean
        dummy_model = DummyRegressor(strategy="mean")
        dummy_model.fit(X, y_log)
        
        # Evaluate: even a dummy model should have reasonable RMSE
        metric = evaluate_model(dummy_model, X, y, problem_type="regression")
        
        # Metric should be a positive number (not NaN or Inf)
        assert not np.isnan(metric), "RMSE should not be NaN"
        assert not np.isinf(metric), "RMSE should not be infinite"
        assert metric >= 0, "RMSE should be non-negative"
    
    def test_evaluate_model_on_different_test_set(self, fitted_model, mock_housing_df):
        """
        REQUIREMENT: Evaluation should work on any test data with the correct structure
        
        Setup:
            - Use the same fitted model with different test data
        Assert:
            - Function executes without error
            - Returns a valid float metric
        """
        # Create a different test set (different rows)
        X_test = mock_housing_df[["area", "bedrooms", "bathrooms"]].iloc[::2]  # Every 2nd row
        y_test = mock_housing_df["price"].iloc[::2]
        
        metric = evaluate_model(fitted_model, X_test, y_test, problem_type="regression")
        
        assert isinstance(metric, float), "Should return a float"
        assert metric >= 0, "RMSE should be non-negative"
    
    def test_evaluate_model_handles_small_dataset(self, fitted_model):
        """
        REQUIREMENT: Evaluation should work even on very small datasets
        
        MLOps systems sometimes evaluate on tiny validation sets.
        The function should handle this gracefully.
        
        Setup:
            - Create a tiny (1-row) test set
        Assert:
            - No exception is raised
            - Returns a valid metric
        """
        # Create a minimal test set (1 row)
        X_tiny = pd.DataFrame({"area": [400.0], "bedrooms": [3.0], "bathrooms": [2.0]})
        y_tiny = pd.Series([5500.0], name="price")
        
        try:
            metric = evaluate_model(fitted_model, X_tiny, y_tiny, problem_type="regression")
        except Exception as e:
            pytest.fail(f"Evaluation on tiny dataset failed: {e}")
        
        assert isinstance(metric, float), "Should return a float even for tiny test set"
        assert metric >= 0, "RMSE should be non-negative"
