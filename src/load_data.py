"""
Educational Goal:
- Why this module exists in an MLOps system: Isolates data ingestion logic from transformation logic, making pipelines testable and source-agnostic
- Responsibility (separation of concerns): This module owns the contract with upstream data providers (files, databases, APIs)
- Pipeline contract (inputs and outputs): Returns a raw DataFrame that subsequent modules can clean and transform predictably

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path
import pandas as pd
from src.utils import load_csv, save_csv


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Inputs:
    - raw_data_path: Path to raw CSV file
    Outputs:
    - df_raw: Raw DataFrame (no transformations applied)
    Why this contract matters for reliable ML delivery:
    - Separating load from clean/transform makes unit testing straightforward
    - Single source of truth for data ingestion reduces debugging surface area when upstream sources change
    """
    print(f"[load_data] Attempting to load raw data from {raw_data_path}")  # TODO: replace with logging later
    
    try:
        df = load_csv(raw_data_path)
        print(f"[load_data] Successfully loaded {len(df)} rows")
        return df
    except FileNotFoundError:
        print("=" * 80)
        print("WARNING: Raw data file not found!")
        print(f"Expected path: {raw_data_path}")
        print("Creating a DUMMY DATASET for scaffolding purposes...")
        print("=" * 80)
        
        # --------------------------------------------------------
        # START STUDENT CODE
        # --------------------------------------------------------
        # TODO_STUDENT: Replace this dummy dataset with your actual data loading logic
        # Why: Real datasets have unique schemas, sources, and access patterns
        # Examples:
        # 1. Load from database: pd.read_sql(query, connection)
        # 2. Load from API: pd.DataFrame(requests.get(url).json())
        # 3. Load from multiple files: pd.concat([pd.read_csv(f) for f in glob('data/*.csv')])
        #
        # Optional forcing function (leave commented)
        # raise NotImplementedError("Student: You must implement this logic to proceed!")
        #
        # Placeholder (Remove this after implementing your code):
        print("Warning: Student has not implemented this section yet")
        # --------------------------------------------------------
        # END STUDENT CODE
        # --------------------------------------------------------
        
        # Baseline: Create deterministic dummy data
        dummy_data = pd.DataFrame({
            "num_feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "cat_feature": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "target": [10.5, 20.3, 15.7, 25.1, 18.9, 28.4, 22.3, 32.1, 26.7, 35.8]
        })
        
        raw_data_path = Path(raw_data_path)
        raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        save_csv(dummy_data, raw_data_path)
        
        print(f"[load_data] Created dummy CSV at {raw_data_path}")
        print("[load_data] IMPORTANT: Update your SETTINGS dictionary in main.py to match your real dataset!")
        print("=" * 80)
        
        return dummy_data