"""
Educational Goal:
- Why this module exists in an MLOps system: Orchestration scripts provide a reproducible, auditable record of the exact steps taken to produce a model
- Responsibility (separation of concerns): This module owns the high-level pipeline flow, calling functions from other modules in the correct sequence
- Pipeline contract (inputs and outputs): Reads configuration, executes end-to-end pipeline, writes artifacts to standardized paths

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path
from sklearn.model_selection import train_test_split

from src.load_data import load_raw_data
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference
from src.utils import save_csv, save_model, load_model


# ============================================================
# CONFIGURATION (will migrate to config.yaml in later session)
# ============================================================
# IMPORTANT: This SETTINGS dictionary is PRE-CONFIGURED for the dummy CSV
# created by load_data.py. When you replace it with your real dataset,
# you MUST update these settings to match your actual columns and problem type!

SETTINGS = {
    "is_example_config": True,  # Set to False after updating for your dataset
    "raw_data_path": "data/raw/dataset.csv",
    "processed_data_path": "data/processed/clean.csv",
    "model_path": "models/model.joblib",
    "predictions_path": "reports/predictions.csv",
    "target_column": "target",
    "problem_type": "regression",  # Options: "regression" or "classification"
    "test_size": 0.2,
    "random_state": 42,
    "features": {
        "quantile_bin": [],  # Example: ["age", "income"] - numeric columns to bin
        "categorical_onehot": ["cat_feature"],  # Example: ["city", "category"]
        "numeric_passthrough": ["num_feature"],  # Example: ["price", "quantity"]
        "n_bins": 3
    }
}


def main():
    """
    Orchestrates the end-to-end ML pipeline.
    
    This function coordinates all pipeline stages in the correct order:
    1. Configuration validation
    2. Data loading and cleaning
    3. Train/test split (BEFORE feature engineering to prevent leakage)
    4. Feature preprocessing recipe creation
    5. Model training with Pipeline
    6. Model evaluation on held-out test set
    7. Inference on example data
    8. Artifact persistence
    """
    
    print("=" * 80)
    print("ML PIPELINE: STARTING END-TO-END EXECUTION")
    print("=" * 80)
    
    # --------------------------------------------------------
    # STEP 1: Ensure output directories exist
    # --------------------------------------------------------
    print("\n[main] Step 1: Creating output directories")
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------------
    # STEP 2: Configuration validation
    # --------------------------------------------------------
    print("\n[main] Step 2: Validating configuration")
    if SETTINGS["is_example_config"]:
        print("=" * 80)
        print("WARNING: You are using the EXAMPLE configuration!")
        print("This is pre-configured to work with the dummy dataset.")
        print("When you paste your real data, you MUST update SETTINGS to match your dataset schema!")
        print("=" * 80)
    
    # --------------------------------------------------------
    # STEP 3: Load raw data
    # --------------------------------------------------------
    print("\n[main] Step 3: Loading raw data")
    df_raw = load_raw_data(Path(SETTINGS["raw_data_path"]))
    
    # --------------------------------------------------------
    # STEP 4: Clean data
    # --------------------------------------------------------
    print("\n[main] Step 4: Cleaning data")
    df_clean = clean_dataframe(df_raw, target_column=SETTINGS["target_column"])
    
    # --------------------------------------------------------
    # STEP 5: Save processed data
    # --------------------------------------------------------
    print("\n[main] Step 5: Saving processed data")
    save_csv(df_clean, Path(SETTINGS["processed_data_path"]))
    
    # --------------------------------------------------------
    # STEP 6: Validate cleaned data
    # --------------------------------------------------------
    print("\n[main] Step 6: Validating cleaned data")
    required_columns = [SETTINGS["target_column"]] + \
                      SETTINGS["features"]["quantile_bin"] + \
                      SETTINGS["features"]["categorical_onehot"] + \
                      SETTINGS["features"]["numeric_passthrough"]
    validate_dataframe(df_clean, required_columns)
    
    # --------------------------------------------------------
    # STEP 7: Train/test split (BEFORE feature engineering)
    # --------------------------------------------------------
    print("\n[main] Step 7: Splitting train/test sets")
    print("[main] CRITICAL: Split happens BEFORE building feature recipe to prevent data leakage")
    
    X = df_clean.drop(columns=[SETTINGS["target_column"]])
    y = df_clean[SETTINGS["target_column"]]
    
    # Use stratification for classification, not for regression
    try:
        if SETTINGS["problem_type"] == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=SETTINGS["test_size"], random_state=SETTINGS["random_state"], stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=SETTINGS["test_size"], random_state=SETTINGS["random_state"]
            )
    except ValueError as e:
        print(f"[main] WARNING: Stratification failed ({e}), falling back to non-stratified split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=SETTINGS["test_size"], random_state=SETTINGS["random_state"]
        )
    
    print(f"[main] Train set: {len(X_train)} samples")
    print(f"[main] Test set: {len(X_test)} samples")
    
    # --------------------------------------------------------
    # STEP 8: Feature engineering sanity checks
    # --------------------------------------------------------
    print("\n[main] Step 8: Validating feature configuration")
    
    # Check all configured columns exist
    all_feature_cols = (SETTINGS["features"]["quantile_bin"] + 
                       SETTINGS["features"]["categorical_onehot"] + 
                       SETTINGS["features"]["numeric_passthrough"])
    missing_cols = set(all_feature_cols) - set(X_train.columns)
    if missing_cols:
        raise ValueError(f"[main] CRITICAL: Configured feature columns do not exist: {missing_cols}")
    
    # Check quantile_bin columns are numeric
    for col in SETTINGS["features"]["quantile_bin"]:
        if X_train[col].dtype not in ['int64', 'float64', 'int32', 'float32']:
            raise TypeError(f"[main] CRITICAL: Column '{col}' configured for quantile_bin but is not numeric (dtype={X_train[col].dtype})")
    
    print("[main] Feature configuration validated")
    
    # --------------------------------------------------------
    # STEP 9: Build feature preprocessing recipe
    # --------------------------------------------------------
    print("\n[main] Step 9: Building feature preprocessing recipe")
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
        categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
        numeric_passthrough_cols=SETTINGS["features"]["numeric_passthrough"],
        n_bins=SETTINGS["features"]["n_bins"]
    )
    
    # --------------------------------------------------------
    # STEP 10: Train model (Pipeline will fit preprocessor + model together)
    # --------------------------------------------------------
    print("\n[main] Step 10: Training model")
    model = train_model(X_train, y_train, preprocessor, problem_type=SETTINGS["problem_type"])
    
    # --------------------------------------------------------
    # STEP 11: Save trained model
    # --------------------------------------------------------
    print("\n[main] Step 11: Saving trained model")
    save_model(model, Path(SETTINGS["model_path"]))
    
    # --------------------------------------------------------
    # STEP 12: Evaluate model on held-out test set
    # --------------------------------------------------------
    print("\n[main] Step 12: Evaluating model")
    metric = evaluate_model(model, X_test, y_test, problem_type=SETTINGS["problem_type"])
    print(f"[main] Test set metric: {metric:.4f}")
    
    # --------------------------------------------------------
    # STEP 13: Run inference on example data (using test set as proxy)
    # --------------------------------------------------------
    print("\n[main] Step 13: Running inference on example data")
    # In production, X_infer would come from new unseen data
    X_infer = X_test.head(10).copy()  # Use first 10 test samples as example
    df_predictions = run_inference(model, X_infer)
    
    # --------------------------------------------------------
    # STEP 14: Save predictions
    # --------------------------------------------------------
    print("\n[main] Step 14: Saving predictions")
    save_csv(df_predictions, Path(SETTINGS["predictions_path"]))
    
    # --------------------------------------------------------
    # PIPELINE COMPLETE
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("ML PIPELINE: EXECUTION COMPLETE")
    print("=" * 80)
    print("\nArtifacts created:")
    print(f"  - Processed data: {SETTINGS['processed_data_path']}")
    print(f"  - Trained model:  {SETTINGS['model_path']}")
    print(f"  - Predictions:    {SETTINGS['predictions_path']}")
    print("\nNext steps:")
    print("  1. Verify artifacts exist and contain expected data")
    print("  2. Replace dummy dataset with your real data")
    print("  3. Update SETTINGS dictionary to match your dataset schema")
    print("  4. Paste your notebook logic into TODO_STUDENT blocks")
    print("=" * 80)


if __name__ == "__main__":
    main()
