"""
Data Preprocessing Module
Cleans, standardizes, and prepares enrollment data for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import re


class DataPreprocessor:
    """
    Handles data cleaning, standardization, and preprocessing
    """

    def __init__(self):
        """Initialize preprocessor"""
        self.date_formats = ["%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning steps to the dataset

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        print(f"\n{'=' * 60}")
        print("PREPROCESSING DATA")
        print(f"{'=' * 60}\n")

        df_clean = df.copy()

        # Step 1: Parse dates
        print("Step 1: Parsing dates...")
        df_clean = self._parse_dates(df_clean)

        # Step 2: Standardize district names
        print("Step 2: Standardizing district names...")
        df_clean = self._standardize_districts(df_clean)

        # Step 3: Validate and clean PIN codes
        print("Step 3: Validating PIN codes...")
        df_clean = self._clean_pincodes(df_clean)

        # Step 4: Handle missing values
        print("Step 4: Handling missing values...")
        df_clean = self._handle_missing_values(df_clean)

        # Step 5: Validate enrollment counts
        print("Step 5: Validating enrollment counts...")
        df_clean = self._validate_enrollments(df_clean)

        # Step 6: Remove duplicates
        print("Step 6: Removing duplicates...")
        df_clean = self._remove_duplicates(df_clean)

        # Step 7: Add derived columns
        print("Step 7: Adding derived columns...")
        df_clean = self._add_derived_columns(df_clean)

        print(f"\nPreprocessing complete!")
        print(f"Original records: {len(df):,}")
        print(f"Cleaned records: {len(df_clean):,}")
        print(f"Records removed: {len(df) - len(df_clean):,}")
        print(f"{'=' * 60}\n")

        return df_clean

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse dates into standard format"""
        if "date" not in df.columns:
            print("  Warning: No 'date' column found")
            return df

        df = df.copy()

        # Try multiple date formats
        best_dates = None
        best_count = 0
        best_fmt = None

        for fmt in self.date_formats:
            try:
                temp_dates = pd.to_datetime(df["date"], format=fmt, errors="coerce")
                valid_count = temp_dates.notna().sum()
                if valid_count > best_count:
                    best_count = valid_count
                    best_dates = temp_dates
                    best_fmt = fmt
            except:
                continue

        if best_dates is not None and best_count > 0:
            print(f"  Successfully parsed {best_count:,} dates using format {best_fmt}")
            df["date"] = best_dates

        # If no format worked, try generic parsing
        if best_dates is None:
            print("  Using generic date parsing...")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Report unparseable dates
        null_dates = df["date"].isna().sum()
        if null_dates > 0:
            print(f"  Warning: {null_dates} dates could not be parsed")

        return df

    def _standardize_districts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize district names"""
        if "district" not in df.columns:
            return df

        df = df.copy()

        # Remove extra whitespace
        df["district"] = df["district"].astype(str).str.strip()

        # Standardize common variations
        district_mapping = {
            "Ranga Reddy": "Rangareddy",
            "RangaReddy": "Rangareddy",
            "Ranga reddy": "Rangareddy",
            "RANGAREDDY": "Rangareddy",
            "HYDERABAD": "Hyderabad",
            "NALGONDA": "Nalgonda",
        }

        df["district"] = df["district"].replace(district_mapping)

        unique_districts = df["district"].unique()
        print(f"  Standardized districts: {', '.join(unique_districts)}")

        return df

    def _clean_pincodes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean PIN codes"""
        if "pincode" not in df.columns:
            return df

        df = df.copy()
        original_count = len(df)

        # Convert to string and remove decimals
        df["pincode"] = df["pincode"].astype(str).str.replace(r"\.0$", "", regex=True)

        # Validate 6-digit format (India PIN codes)
        df["pincode_valid"] = df["pincode"].str.match(r"^\d{6}$")

        invalid_count = (~df["pincode_valid"]).sum()
        if invalid_count > 0:
            print(f"  Warning: {invalid_count} invalid PIN codes found")
            # Keep invalid PIN codes but flag them

        # Convert to integer where valid
        df.loc[df["pincode_valid"], "pincode"] = df.loc[
            df["pincode_valid"], "pincode"
        ].astype(int)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in enrollment columns"""
        df = df.copy()

        enrollment_cols = ["age_0_5", "age_5_17", "age_18_greater"]

        for col in enrollment_cols:
            if col in df.columns:
                # Fill missing enrollment values with 0 (legitimate zero enrollment)
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    print(f"  Filling {missing_count} missing values in {col} with 0")
                    df[col] = df[col].fillna(0)

                # Ensure non-negative
                df[col] = df[col].clip(lower=0)

        return df

    def _validate_enrollments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate enrollment counts"""
        df = df.copy()

        enrollment_cols = ["age_0_5", "age_5_17", "age_18_greater"]

        # Check for unreasonably high values (potential data errors)
        max_reasonable = 10000  # Maximum reasonable enrollments per PIN per month

        for col in enrollment_cols:
            if col in df.columns:
                high_values = (df[col] > max_reasonable).sum()
                if high_values > 0:
                    print(
                        f"  Warning: {high_values} records with {col} > {max_reasonable}"
                    )

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove or aggregate duplicate records"""
        df = df.copy()
        original_count = len(df)

        # Check for duplicates based on date, district, pincode
        key_cols = ["date", "district", "pincode"]
        existing_cols = [col for col in key_cols if col in df.columns]

        if len(existing_cols) < len(key_cols):
            print(f"  Cannot check duplicates: missing columns")
            return df

        duplicates = df.duplicated(subset=existing_cols, keep=False)
        dup_count = duplicates.sum()

        if dup_count > 0:
            print(f"  Found {dup_count} duplicate records")
            print(f"  Aggregating duplicates by summing enrollments...")

            # Aggregate duplicates by summing enrollments
            enrollment_cols = ["age_0_5", "age_5_17", "age_18_greater"]
            agg_dict = {col: "sum" for col in enrollment_cols if col in df.columns}

            # Keep first occurrence of other columns
            for col in df.columns:
                if col not in existing_cols and col not in enrollment_cols:
                    agg_dict[col] = "first"

            df = df.groupby(existing_cols, as_index=False).agg(agg_dict)
            print(f"  Reduced to {len(df):,} records after aggregation")
        else:
            print(f"  No duplicates found ✓")

        return df

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful derived columns"""
        df = df.copy()

        # Total enrollments
        enrollment_cols = ["age_0_5", "age_5_17", "age_18_greater"]
        if all(col in df.columns for col in enrollment_cols):
            df["total_enrollments"] = df[enrollment_cols].sum(axis=1)
            print(f"  Added 'total_enrollments' column")

        # Extract month, year from date
        if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["month_name"] = df["date"].dt.strftime("%B")
            df["year_month"] = df["date"].dt.to_period("M").astype(str)
            print(f"  Added date-related columns (year, month, month_name, year_month)")

        return df

    def aggregate_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data by month, district, and PIN code

        Args:
            df: Cleaned DataFrame

        Returns:
            Monthly aggregated DataFrame
        """
        if "date" not in df.columns:
            print("Cannot aggregate: 'date' column missing")
            return df

        print(f"\nAggregating data by month...")

        group_cols = ["year_month", "district", "pincode"]
        enrollment_cols = ["age_0_5", "age_5_17", "age_18_greater"]

        agg_dict = {col: "sum" for col in enrollment_cols if col in df.columns}
        agg_dict["total_enrollments"] = "sum"

        df_monthly = df.groupby(group_cols, as_index=False).agg(agg_dict)

        print(f"  Aggregated to {len(df_monthly):,} monthly records")

        return df_monthly


def preprocess_aadhaar_data(
    df: pd.DataFrame, aggregate_monthly: bool = False
) -> pd.DataFrame:
    """
    Convenience function to preprocess Aadhaar enrollment data

    Args:
        df: Raw DataFrame
        aggregate_monthly: Whether to aggregate by month

    Returns:
        Preprocessed DataFrame
    """
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)

    if aggregate_monthly:
        df_clean = preprocessor.aggregate_monthly(df_clean)

    return df_clean


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import load_aadhaar_data

    df = load_aadhaar_data("../data/raw")
    df_clean = preprocess_aadhaar_data(df, aggregate_monthly=True)

    print(f"\nCleaned data shape: {df_clean.shape}")
    print(f"\nSample data:")
    print(df_clean.head(10))
