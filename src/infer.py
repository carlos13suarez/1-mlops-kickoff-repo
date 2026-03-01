"""
Educational Goal:
- Why this module exists in an MLOps system: Production inference must use the exact same preprocessing as training to prevent train/serve skew
- Responsibility (separation of concerns): This module owns the contract for applying trained models to new data
- Pipeline contract (inputs and outputs): Accepts FITTED Pipeline and new data, returns predictions in a standardized format for downstream systems

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model: FITTED Pipeline from train.py (loaded via utils.load_model)
    - X_infer: New data for prediction (DataFrame with same schema as training data)
    Outputs:
    - df_predictions: DataFrame with single column "prediction" (preserves input index)
    Why this contract matters for reliable ML delivery:
    - Pipeline ensures preprocessing is applied identically to training (prevents train/serve skew)
    - Preserving index allows downstream systems to join predictions back to original records
    - Single-column output format simplifies integration with databases and APIs
    """
    print(f"[infer] Running inference on {len(X_infer)} samples")  # TODO: replace with logging later
    
    # Generate predictions using full Pipeline (preprocess + predict)
    predictions = model.predict(X_infer)
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add post-processing logic for production requirements
    # Why: Production systems often need additional transformations (probability scores, business rules, formatting)
    # Examples:
    # 1. Return probabilities instead of classes:
    #    predictions = model.predict_proba(X_infer)[:, 1]  # Probability of positive class
    # 2. Apply business rules:
    #    predictions = np.where(predictions < min_threshold, min_threshold, predictions)
    # 3. Add confidence scores:
    #    df_predictions['confidence'] = model.predict_proba(X_infer).max(axis=1)
    # 4. Round predictions:
    #    predictions = np.round(predictions, 2)
    # 5. Map predictions to labels:
    #    label_map = {0: 'low', 1: 'medium', 2: 'high'}
    #    predictions = [label_map[p] for p in predictions]
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    # Return DataFrame with single column (preserving input index)
    df_predictions = pd.DataFrame({"prediction": predictions}, index=X_infer.index)
    
    print(f"[infer] Inference complete: {len(df_predictions)} predictions generated")
    return df_predictions