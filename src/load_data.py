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
import kagglehub


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
        # Check if the file already exists locally to avoid redundant downloads
        df = load_csv(raw_data_path)
        print(f"[load_data] Successfully loaded {len(df)} rows from local cache")
        return df
    except FileNotFoundError:
        print("=" * 80)
        print(f"WARNING: Raw data file not found at {raw_data_path}. Initializing ingestion...")

        # --------------------------------------------------------
        # START STUDENT CODE
        # --------------------------------------------------------
        try:
            print("Fetching 'yasserh/housing-prices-dataset' via kagglehub...")
            # Download the latest version
            download_path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
            
            # Locate the specific CSV in the downloaded folder
            downloaded_file = Path(download_path) / "Housing.csv"
            df = pd.read_csv(downloaded_file)
            
            # Save it to the project's data/raw folder for pipeline consistency
            save_csv(df, raw_data_path)
            print(f"Successfully ingested {len(df)} rows into {raw_data_path}")
            return df
            
        except Exception as e:
            print(f"Failed to ingest from Kaggle: {e}")
            print("Falling back to dummy baseline to keep pipeline runnable.")
            # --------------------------------------------------------
            # END STUDENT CODE
            # --------------------------------------------------------
            
            # Baseline: Create deterministic dummy data matching the Housing schema
            print("LOUD WARNING: CREATING DUMMY DATASET FOR SCAFFOLDING ONLY. UPDATE SETTINGS.")
            dummy_data = pd.DataFrame({
                "area": [2000, 3000, 4000],
                "bedrooms": [2, 3, 4],
                "bathrooms": [1, 2, 2],
                "stories": [1, 2, 2],
                "mainroad": ["yes", "yes", "no"],
                "guestroom": ["no", "no", "yes"],
                "basement": ["no", "yes", "no"],
                "hotwaterheating": ["no", "no", "no"],
                "airconditioning": ["yes", "yes", "no"],
                "parking": [1, 2, 2],
                "prefarea": ["yes", "no", "no"],
                "furnishingstatus": ["furnished", "semi-furnished", "unfurnished"],
                "price": [500000.0, 750000.0, 900000.0]
            })
            
            raw_data_path = Path(raw_data_path)
            raw_data_path.parent.mkdir(parents=True, exist_ok=True)
            save_csv(dummy_data, raw_data_path)
            print(f"[load_data] Created dummy CSV at {raw_data_path}")
            print("=" * 80)
            
            return dummy_data