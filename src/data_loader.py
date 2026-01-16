"""
Data Loading Module
Handles loading and initial validation of Aadhaar enrollment data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class DataLoader:
    """
    Loads and validates Aadhaar enrollment data from CSV files
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize DataLoader

        Args:
            data_dir: Directory containing raw CSV files
        """
        self.data_dir = Path(data_dir)
        self.required_columns = [
            "date",
            "state",
            "district",
            "pincode",
            "age_0_5",
            "age_5_17",
            "age_18_greater",
        ]

    def load_single_file(self, filename: str) -> pd.DataFrame:
        """
        Load a single CSV file with validation

        Args:
            filename: Name of CSV file to load

        Returns:
            DataFrame with loaded data
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        print(f"Loading {filename}...")

        try:
            df = pd.read_csv(filepath)
            print(f"  Loaded {len(df)} records")

            # Validate columns
            missing_cols = set(self.required_columns) - set(df.columns)
            if missing_cols:
                print(f"  Warning: Missing columns {missing_cols}")

            return df

        except Exception as e:
            print(f"  Error loading {filename}: {str(e)}")
            raise

    def load_all_files(self, filenames: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load and combine multiple CSV files

        Args:
            filenames: List of filenames to load. If None, loads all CSV files

        Returns:
            Combined DataFrame
        """
        if filenames is None:
            # Auto-detect CSV files
            filenames = [f.name for f in self.data_dir.glob("*.csv")]

        if not filenames:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        print(f"\n{'=' * 60}")
        print(f"Loading {len(filenames)} data files")
        print(f"{'=' * 60}\n")

        dataframes = []

        for filename in filenames:
            try:
                df = self.load_single_file(filename)
                dataframes.append(df)
            except Exception as e:
                print(f"Skipping {filename} due to error: {str(e)}")

        if not dataframes:
            raise ValueError("No valid data files could be loaded")

        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)

        print(f"\n{'=' * 60}")
        print(f"Total records loaded: {len(combined_df):,}")
        print(f"{'=' * 60}\n")

        return combined_df

    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate loaded data and return summary statistics

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        validation_report = {
            "total_records": len(df),
            "columns_present": list(df.columns),
            "missing_required_columns": list(
                set(self.required_columns) - set(df.columns)
            ),
            "date_range": None,
            "districts": [],
            "pincodes": [],
            "missing_values": {},
            "data_types": {},
        }

        # Check for missing required columns
        if validation_report["missing_required_columns"]:
            print(
                f"Warning: Missing required columns: {validation_report['missing_required_columns']}"
            )
            return validation_report

        # Date range
        if "date" in df.columns:
            try:
                dates = pd.to_datetime(df["date"], errors="coerce")
                validation_report["date_range"] = (dates.min(), dates.max())
            except:
                validation_report["date_range"] = "Could not parse dates"

        # Geographic coverage
        if "district" in df.columns:
            validation_report["districts"] = df["district"].unique().tolist()

        if "pincode" in df.columns:
            validation_report["pincodes"] = df["pincode"].nunique()

        # Missing values
        validation_report["missing_values"] = (
            df[self.required_columns].isnull().sum().to_dict()
        )

        # Data types
        validation_report["data_types"] = (
            df[self.required_columns].dtypes.astype(str).to_dict()
        )

        # Print validation summary
        self._print_validation_summary(validation_report)

        return validation_report

    def _print_validation_summary(self, report: Dict):
        """Print formatted validation summary"""
        print(f"\n{'=' * 60}")
        print("DATA VALIDATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total Records: {report['total_records']:,}")

        if report["date_range"] and isinstance(report["date_range"], tuple):
            print(f"Date Range: {report['date_range'][0]} to {report['date_range'][1]}")

        print(f"Districts: {len(report['districts'])}")
        for district in report["districts"]:
            print(f"  - {district}")

        print(f"Unique PIN Codes: {report['pincodes']:,}")

        print(f"\nMissing Values:")
        for col, count in report["missing_values"].items():
            if count > 0:
                print(f"  {col}: {count}")

        if not any(report["missing_values"].values()):
            print("  None ✓")

        print(f"{'=' * 60}\n")


def load_aadhaar_data(
    data_dir: str = "data/raw", filenames: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convenience function to load Aadhaar enrollment data

    Args:
        data_dir: Directory containing raw CSV files
        filenames: Optional list of specific files to load

    Returns:
        Combined DataFrame with all enrollment data
    """
    loader = DataLoader(data_dir)
    df = loader.load_all_files(filenames)
    loader.validate_data(df)

    return df


if __name__ == "__main__":
    # Test the data loader
    df = load_aadhaar_data("../data/raw")
    print(f"\nLoaded data shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
