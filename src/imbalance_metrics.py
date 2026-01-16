"""
Enrollment Imbalance Metrics Module
Calculates custom metrics to quantify age-wise enrollment imbalances
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class ImbalanceMetrics:
    """
    Calculate enrollment imbalance metrics for Aadhaar data
    """

    def __init__(self):
        """Initialize metrics calculator"""
        self.metrics_calculated = []

    def calculate_aer(
        self, age_0_5: float, age_5_17: float, age_18_greater: float
    ) -> float:
        """
        Calculate Adult Enrollment Ratio (AER)

        Formula: AER = age_18_greater / (age_0_5 + age_5_17 + age_18_greater)

        Args:
            age_0_5: Enrollments in 0-5 age group
            age_5_17: Enrollments in 5-17 age group
            age_18_greater: Enrollments in 18+ age group

        Returns:
            Adult Enrollment Ratio (0 to 1)
        """
        total = age_0_5 + age_5_17 + age_18_greater

        if total == 0:
            return 0.0

        aer = age_18_greater / total
        return aer

    def calculate_caes(self, age_0_5: float, age_18_greater: float) -> float:
        """
        Calculate Child-to-Adult Enrollment Skew (CAES)

        Formula: CAES = age_0_5 / (age_18_greater + 1)

        Args:
            age_0_5: Enrollments in 0-5 age group
            age_18_greater: Enrollments in 18+ age group

        Returns:
            Child-to-Adult ratio
        """
        # Add 1 to avoid division by zero
        caes = age_0_5 / (age_18_greater + 1)
        return caes

    def calculate_aebi(
        self, age_0_5: float, age_5_17: float, age_18_greater: float
    ) -> float:
        """
        Calculate Age Enrollment Balance Index (AEBI) - Composite Metric

        Formula: AEBI = (AER × 100) + (1 - |ratio_5_17 - 0.05| / 0.1)
        where ratio_5_17 = age_5_17 / total

        Args:
            age_0_5: Enrollments in 0-5 age group
            age_5_17: Enrollments in 5-17 age group
            age_18_greater: Enrollments in 18+ age group

        Returns:
            Age Enrollment Balance Index (0 to 100)
        """
        total = age_0_5 + age_5_17 + age_18_greater

        if total == 0:
            return 0.0

        # Calculate AER component
        aer = self.calculate_aer(age_0_5, age_5_17, age_18_greater)

        # Calculate 5-17 ratio component
        ratio_5_17 = age_5_17 / total

        # AEBI formula
        aebi = (aer * 100) + (1 - abs(ratio_5_17 - 0.05) / 0.1)

        # Clamp to 0-100 range
        aebi = max(0, min(100, aebi))

        return aebi

    def classify_aer_grade(self, aer: float) -> str:
        """
        Classify AER into performance grades

        Args:
            aer: Adult Enrollment Ratio

        Returns:
            Grade string (GREEN/YELLOW/RED)
        """
        if aer >= 0.05:  # 5% or higher
            return "GREEN"
        elif aer >= 0.01:  # 1-5%
            return "YELLOW"
        else:  # Less than 1%
            return "RED"

    def classify_aebi_grade(self, aebi: float) -> str:
        """
        Classify AEBI into performance grades

        Args:
            aebi: Age Enrollment Balance Index

        Returns:
            Grade string (GREEN/YELLOW/RED)
        """
        if aebi >= 7:
            return "GREEN"
        elif aebi >= 3:
            return "YELLOW"
        else:
            return "RED"

    def add_metrics_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all metrics as new columns to DataFrame

        Args:
            df: DataFrame with age group enrollment columns

        Returns:
            DataFrame with added metric columns
        """
        print(f"\n{'=' * 60}")
        print("CALCULATING ENROLLMENT IMBALANCE METRICS")
        print(f"{'=' * 60}\n")

        df = df.copy()

        required_cols = ["age_0_5", "age_5_17", "age_18_greater"]

        # Check for required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Ensure total_enrollments exists for percentage calculations
        if "total_enrollments" not in df.columns:
            df["total_enrollments"] = df[required_cols].sum(axis=1)

        # Calculate AER
        print("Calculating Adult Enrollment Ratio (AER)...")
        df["aer"] = df.apply(
            lambda row: self.calculate_aer(
                row["age_0_5"], row["age_5_17"], row["age_18_greater"]
            ),
            axis=1,
        )

        # Calculate CAES
        print("Calculating Child-to-Adult Enrollment Skew (CAES)...")
        df["caes"] = df.apply(
            lambda row: self.calculate_caes(row["age_0_5"], row["age_18_greater"]),
            axis=1,
        )

        # Calculate AEBI
        print("Calculating Age Enrollment Balance Index (AEBI)...")
        df["aebi"] = df.apply(
            lambda row: self.calculate_aebi(
                row["age_0_5"], row["age_5_17"], row["age_18_greater"]
            ),
            axis=1,
        )

        # Add percentage columns for readability
        df["aer_pct"] = df["aer"] * 100
        df["child_pct"] = (df["age_0_5"] / df["total_enrollments"] * 100).fillna(0)
        df["youth_pct"] = (df["age_5_17"] / df["total_enrollments"] * 100).fillna(0)
        df["adult_pct"] = (df["age_18_greater"] / df["total_enrollments"] * 100).fillna(
            0
        )

        # Add grade classifications
        print("Classifying performance grades...")
        df["aer_grade"] = df["aer"].apply(self.classify_aer_grade)
        df["aebi_grade"] = df["aebi"].apply(self.classify_aebi_grade)

        # Add priority ranking (lower is higher priority)
        df["priority_score"] = df["aer"].rank(ascending=True)

        print(f"\nMetrics calculation complete!")
        print(f"Added columns: aer, caes, aebi, aer_pct, *_pct, grades")
        print(f"{'=' * 60}\n")

        self.metrics_calculated = [
            "aer",
            "caes",
            "aebi",
            "aer_pct",
            "aer_grade",
            "aebi_grade",
            "priority_score",
        ]

        return df

    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics for metrics

        Args:
            df: DataFrame with calculated metrics

        Returns:
            Dictionary with summary statistics
        """
        if "aer" not in df.columns:
            raise ValueError(
                "Metrics not calculated. Run add_metrics_to_dataframe first."
            )

        summary = {
            "aer": {
                "mean": df["aer"].mean(),
                "median": df["aer"].median(),
                "std": df["aer"].std(),
                "min": df["aer"].min(),
                "max": df["aer"].max(),
                "q25": df["aer"].quantile(0.25),
                "q75": df["aer"].quantile(0.75),
            },
            "aebi": {
                "mean": df["aebi"].mean(),
                "median": df["aebi"].median(),
                "std": df["aebi"].std(),
                "min": df["aebi"].min(),
                "max": df["aebi"].max(),
            },
            "grade_distribution": {
                "aer": df["aer_grade"].value_counts().to_dict(),
                "aebi": df["aebi_grade"].value_counts().to_dict(),
            },
            "enrollment_distribution": {
                "avg_child_pct": df["child_pct"].mean(),
                "avg_youth_pct": df["youth_pct"].mean(),
                "avg_adult_pct": df["adult_pct"].mean(),
            },
        }

        return summary

    def print_summary(self, df: pd.DataFrame):
        """Print formatted summary of metrics"""
        summary = self.get_summary_statistics(df)

        print(f"\n{'=' * 60}")
        print("ENROLLMENT IMBALANCE METRICS SUMMARY")
        print(f"{'=' * 60}\n")

        print("Adult Enrollment Ratio (AER):")
        print(
            f"  Mean:   {summary['aer']['mean']:.4f} ({summary['aer']['mean'] * 100:.2f}%)"
        )
        print(
            f"  Median: {summary['aer']['median']:.4f} ({summary['aer']['median'] * 100:.2f}%)"
        )
        print(f"  Range:  {summary['aer']['min']:.4f} - {summary['aer']['max']:.4f}")
        print(f"  Std:    {summary['aer']['std']:.4f}")

        print(f"\nAge Enrollment Balance Index (AEBI):")
        print(f"  Mean:   {summary['aebi']['mean']:.2f}")
        print(f"  Median: {summary['aebi']['median']:.2f}")
        print(f"  Range:  {summary['aebi']['min']:.2f} - {summary['aebi']['max']:.2f}")

        print(f"\nPerformance Grade Distribution (AER):")
        for grade, count in sorted(summary["grade_distribution"]["aer"].items()):
            pct = count / len(df) * 100
            print(f"  {grade:6s}: {count:5d} ({pct:5.1f}%)")

        print(f"\nAverage Enrollment Distribution:")
        print(
            f"  Children (0-5):   {summary['enrollment_distribution']['avg_child_pct']:.1f}%"
        )
        print(
            f"  Youth (5-17):     {summary['enrollment_distribution']['avg_youth_pct']:.1f}%"
        )
        print(
            f"  Adults (18+):     {summary['enrollment_distribution']['avg_adult_pct']:.1f}%"
        )

        print(f"{'=' * 60}\n")


def calculate_metrics(df: pd.DataFrame, print_summary: bool = True) -> pd.DataFrame:
    """
    Convenience function to calculate all metrics

    Args:
        df: DataFrame with enrollment data
        print_summary: Whether to print summary statistics

    Returns:
        DataFrame with added metric columns
    """
    metrics = ImbalanceMetrics()
    df_with_metrics = metrics.add_metrics_to_dataframe(df)

    if print_summary:
        metrics.print_summary(df_with_metrics)

    return df_with_metrics


if __name__ == "__main__":
    # Test metrics calculation
    from data_loader import load_aadhaar_data
    from preprocessing import preprocess_aadhaar_data

    df = load_aadhaar_data("../data/raw")
    df_clean = preprocess_aadhaar_data(df, aggregate_monthly=True)
    df_metrics = calculate_metrics(df_clean)

    print("\nSample data with metrics:")
    print(
        df_metrics[
            ["district", "pincode", "total_enrollments", "aer_pct", "aebi", "aer_grade"]
        ].head(10)
    )
