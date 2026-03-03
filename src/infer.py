"""
Educational Goal:
- Why this module exists in an MLOps system: Production inference must use the exact same preprocessing as training to prevent train/serve skew
- Responsibility (separation of concerns): This module owns the contract for applying trained models to new data
- Pipeline contract (inputs and outputs): Accepts FITTED Pipeline and new data, returns predictions in a standardized format for downstream systems

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd
import numpy as np

def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model: FITTED Pipeline from train.py (includes preprocessor)
    - X_infer: New data for prediction (DataFrame)
    Outputs:
    - df_predictions: DataFrame with final dollar predictions
    Why this contract matters for reliable ML delivery:
    - Pipeline ensures preprocessing is applied identically to training (prevents train/serve skew)
    - Preserving index allows downstream systems to join predictions back to original records
    - Single-column output format simplifies integration with databases and APIs
    """
    print(f"[infer] Running inference on {len(X_infer)} samples")  # TODO: replace with logging later
    
    # 1. Generate predictions using full Pipeline
    # The Pipeline automatically applies the binning/scaling from features.py
    # These predictions are in LOG space because of our train.py logic
    y_pred_log = model.predict(X_infer)
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    
    # Convert log-space predictions back to original scale (exponentiate)
    predictions = np.expm1(y_pred_log)

    # Round predictions to 2 decimal places for production
    predictions = np.round(predictions, 2)

    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    # Return DataFrame with single column (preserving input index)
    df_predictions = pd.DataFrame({"prediction": predictions}, index=X_infer.index)
    
    print(f"[infer] Inference complete: {len(df_predictions)} predictions generated")
    return df_predictions