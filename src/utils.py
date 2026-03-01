"""
Educational Goal:
- Why this module exists in an MLOps system: Centralize I/O operations to reduce file handling errors and ensure consistent serialization across pipeline stages
- Responsibility (separation of concerns): This module owns all interactions with the filesystem (CSV and model persistence)
- Pipeline contract (inputs and outputs): Provides reusable save/load functions that guarantee reproducibility and compatibility

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path
import pandas as pd
import joblib


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: Absolute or relative path to CSV file
    Outputs:
    - DataFrame loaded from CSV
    Why this contract matters for reliable ML delivery:
    - Centralized loading prevents silent failures and encoding mismatches across team members
    - Path validation catches missing data issues early
    """
    print(f"[utils] Loading CSV from {filepath}")  # TODO: replace with logging later
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Customize CSV loading parameters for your dataset
    # Why: Different datasets require different delimiters, encodings, date parsers
    # Examples:
    # 1. df = pd.read_csv(filepath, sep=';', encoding='latin1')
    # 2. df = pd.read_csv(filepath, parse_dates=['timestamp'])
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    df = pd.read_csv(filepath)
    return df


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df: DataFrame to save
    - filepath: Destination path
    Outputs:
    - None (side effect: writes CSV to disk)
    Why this contract matters for reliable ML delivery:
    - Automatically creates output directories, preventing pipeline crashes from missing folders
    - Enforces index=False by default to avoid serialization bugs downstream
    """
    print(f"[utils] Saving CSV to {filepath}")  # TODO: replace with logging later
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Customize CSV saving parameters if needed
    # Why: Some systems require specific formats (no index, specific encoding, compression)
    # Examples:
    # 1. df.to_csv(filepath, index=False, compression='gzip')
    # 2. df.to_csv(filepath, index=False, encoding='utf-8-sig')
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """
    Inputs:
    - model: Trained scikit-learn model or Pipeline
    - filepath: Destination path for .joblib file
    Outputs:
    - None (side effect: serializes model to disk)
    Why this contract matters for reliable ML delivery:
    - Version-controlled model artifacts enable rollback and A/B testing
    - Automatic directory creation ensures deployment scripts never fail on missing folders
    """
    print(f"[utils] Saving model to {filepath}")  # TODO: replace with logging later
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Switch to alternative serialization if needed
    # Why: Some models (e.g., deep learning) require framework-specific formats
    # Examples:
    # 1. joblib.dump(model, filepath, compress=3)  # Higher compression
    # 2. Use pickle.dump() or torch.save() for non-sklearn models
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """
    Inputs:
    - filepath: Path to serialized model
    Outputs:
    - Loaded model object (typically a scikit-learn Pipeline)
    Why this contract matters for reliable ML delivery:
    - Guarantees loaded models match training environment serialization format
    - Centralizes model loading to detect corruption or version mismatches early
    """
    print(f"[utils] Loading model from {filepath}")  # TODO: replace with logging later
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Match deserialization method to your save_model implementation
    # Why: Serialization and deserialization must use compatible formats
    # Examples:
    # 1. model = joblib.load(filepath)
    # 2. model = pickle.load(open(filepath, 'rb'))
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    model = joblib.load(filepath)
    return model