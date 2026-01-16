"""
Ranking Engine Module
Ranks districts and PIN codes by enrollment performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class RankingEngine:
    """
    Ranks geographic regions by enrollment imbalance metrics
    """
    
    def __init__(self):
        """Initialize ranking engine"""
        self.district_rankings = None
        self.pincode_rankings = None
    
    def rank_districts(self, df: pd.DataFrame, 
                       metric: str = 'aer') -> pd.DataFrame:
        """
        Rank districts by enrollment performance
        
        Args:
            df: DataFrame with calculated metrics
            metric: Metric to rank by ('aer', 'aebi', etc.)
            
        Returns:
            DataFrame with district rankings
        """
        print(f"\n{'='*60}")
        print("RANKING DISTRICTS")
        print(f"{'='*60}\n")
        
        # Aggregate by district
        agg_dict = {
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum',
            'total_enrollments': 'sum'
        }
        
        district_data = df.groupby('district').agg(agg_dict).reset_index()
        
        # Recalculate metrics at district level
        from imbalance_metrics import ImbalanceMetrics
        metrics_calc = ImbalanceMetrics()
        district_data = metrics_calc.add_metrics_to_dataframe(district_data)
        
        # Sort by metric (lower AER = worse performance = higher priority)
        district_data = district_data.sort_values(metric, ascending=True)
        
        # Add ranking
        district_data['rank'] = range(1, len(district_data) + 1)
        district_data['priority'] = district_data['rank'].apply(
            lambda x: 'HIGH' if x <= len(district_data) * 0.3 
                     else 'MEDIUM' if x <= len(district_data) * 0.7 
                     else 'LOW'
        )
        
        self.district_rankings = district_data
        
        print(f"Ranked {len(district_data)} districts by {metric.upper()}")
        print(f"{'='*60}\n")
        
        return district_data
    
    def rank_pincodes(self, df: pd.DataFrame, 
                      metric: str = 'aer',
                      min_enrollments: int = 10,
                      top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Rank PIN codes by enrollment performance
        
        Args:
            df: DataFrame with calculated metrics
            metric: Metric to rank by
            min_enrollments: Minimum total enrollments to include
            top_n: Return only top N PIN codes (None = all)
            
        Returns:
            DataFrame with PIN code rankings
        """
        print(f"\n{'='*60}")
        print("RANKING PIN CODES")
        print(f"{'='*60}\n")
        
        # Aggregate by PIN code and district
        agg_dict = {
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum',
            'total_enrollments': 'sum'
        }
        
        pincode_data = df.groupby(['district', 'pincode']).agg(agg_dict).reset_index()
        
        # Filter by minimum enrollments
        pincode_data = pincode_data[pincode_data['total_enrollments'] >= min_enrollments]
        
        # Recalculate metrics
        from imbalance_metrics import ImbalanceMetrics
        metrics_calc = ImbalanceMetrics()
        pincode_data = metrics_calc.add_metrics_to_dataframe(pincode_data)
        
        # Sort by metric
        pincode_data = pincode_data.sort_values(metric, ascending=True)
        
        # Add ranking
        pincode_data['rank'] = range(1, len(pincode_data) + 1)
        
        # Classify risk level
        pincode_data['risk_level'] = pincode_data.apply(
            lambda row: 'HIGH' if row['aer'] < 0.005 
                       else 'MEDIUM' if row['aer'] < 0.02 
                       else 'LOW',
            axis=1
        )
        
        # Add recommendation
        pincode_data['recommendation'] = pincode_data.apply(
            self._generate_recommendation, axis=1
        )
        
        self.pincode_rankings = pincode_data
        
        if top_n:
            pincode_data = pincode_data.head(top_n)
        
        print(f"Ranked {len(pincode_data)} PIN codes by {metric.upper()}")
        print(f"(Filtered: minimum {min_enrollments} enrollments)")
        print(f"{'='*60}\n")
        
        return pincode_data
    
    def _generate_recommendation(self, row: pd.Series) -> str:
        """Generate recommendation based on metrics"""
        if row['age_18_greater'] == 0:
            return "Targeted adult drive needed"
        elif row['aer'] < 0.01:
            return "Adult outreach required"
        elif row['aer'] < 0.03:
            return "Moderate improvement needed"
        elif row['aer'] < 0.05:
            return "Slight optimization possible"
        else:
            return "Best-practice center"
    
    def identify_priority_zones(self, df: pd.DataFrame,
                                aer_threshold: float = 0.01,
                                min_enrollments: int = 50) -> pd.DataFrame:
        """
        Identify high-priority intervention zones
        
        Args:
            df: DataFrame with metrics
            aer_threshold: AER threshold for priority (lower = priority)
            min_enrollments: Minimum enrollments to be considered
            
        Returns:
            DataFrame with priority zones
        """
        print(f"\n{'='*60}")
        print("IDENTIFYING PRIORITY INTERVENTION ZONES")
        print(f"{'='*60}\n")
        
        # Aggregate by PIN code
        agg_dict = {
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum',
            'total_enrollments': 'sum'
        }
        
        pincode_data = df.groupby(['district', 'pincode']).agg(agg_dict).reset_index()
        
        # Filter
        pincode_data = pincode_data[pincode_data['total_enrollments'] >= min_enrollments]
        
        # Recalculate metrics
        from imbalance_metrics import ImbalanceMetrics
        metrics_calc = ImbalanceMetrics()
        pincode_data = metrics_calc.add_metrics_to_dataframe(pincode_data)
        
        # Identify priority zones
        priority_zones = pincode_data[pincode_data['aer'] < aer_threshold].copy()
        priority_zones = priority_zones.sort_values('aer', ascending=True)
        
        # Add tier classification
        priority_zones['tier'] = priority_zones.apply(
            lambda row: 'TIER 1 - URGENT' if row['aer'] < 0.001
                       else 'TIER 2 - MODERATE' if row['aer'] < 0.005
                       else 'TIER 3 - LOW PRIORITY',
            axis=1
        )
        
        print(f"Identified {len(priority_zones)} priority zones")
        print(f"  (AER < {aer_threshold*100:.1f}%, min enrollments: {min_enrollments})")
        
        # Print tier breakdown
        tier_counts = priority_zones['tier'].value_counts()
        for tier in ['TIER 1 - URGENT', 'TIER 2 - MODERATE', 'TIER 3 - LOW PRIORITY']:
            if tier in tier_counts.index:
                print(f"  {tier}: {tier_counts[tier]} zones")
        
        print(f"{'='*60}\n")
        
        return priority_zones
    
    def analyze_within_district_variation(self, df: pd.DataFrame,
                                          min_pincodes: int = 5) -> pd.DataFrame:
        """
        Analyze variation in enrollment performance within districts
        
        Args:
            df: DataFrame with metrics
            min_pincodes: Minimum PIN codes for district to be analyzed
            
        Returns:
            DataFrame with within-district variation statistics
        """
        print(f"\n{'='*60}")
        print("ANALYZING WITHIN-DISTRICT VARIATION")
        print(f"{'='*60}\n")
        
        # Aggregate by PIN code
        pincode_data = df.groupby(['district', 'pincode']).agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum',
            'total_enrollments': 'sum'
        }).reset_index()
        
        # Calculate metrics
        from imbalance_metrics import ImbalanceMetrics
        metrics_calc = ImbalanceMetrics()
        pincode_data = metrics_calc.add_metrics_to_dataframe(pincode_data)
        
        # Calculate variation statistics by district
        variation_stats = []
        
        for district in pincode_data['district'].unique():
            district_pins = pincode_data[pincode_data['district'] == district]
            
            if len(district_pins) < min_pincodes:
                continue
            
            stats = {
                'district': district,
                'num_pincodes': len(district_pins),
                'mean_aer': district_pins['aer'].mean(),
                'median_aer': district_pins['aer'].median(),
                'std_aer': district_pins['aer'].std(),
                'min_aer': district_pins['aer'].min(),
                'max_aer': district_pins['aer'].max(),
                'cv_aer': (district_pins['aer'].std() / district_pins['aer'].mean() * 100 
                          if district_pins['aer'].mean() > 0 else 0),
                'range_aer': district_pins['aer'].max() - district_pins['aer'].min(),
                'best_pincode': district_pins.loc[district_pins['aer'].idxmax(), 'pincode'],
                'worst_pincode': district_pins.loc[district_pins['aer'].idxmin(), 'pincode']
            }
            
            variation_stats.append(stats)
        
        variation_df = pd.DataFrame(variation_stats)
        variation_df = variation_df.sort_values('cv_aer', ascending=False)
        
        print(f"Analyzed {len(variation_df)} districts")
        print(f"{'='*60}\n")
        
        return variation_df
    
    def create_performance_scorecard(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive performance scorecard for districts
        
        Args:
            df: DataFrame with metrics
            
        Returns:
            Scorecard DataFrame
        """
        print(f"\n{'='*60}")
        print("CREATING DISTRICT PERFORMANCE SCORECARD")
        print(f"{'='*60}\n")
        
        # Aggregate by district
        scorecard = df.groupby('district').agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum',
            'total_enrollments': 'sum'
        }).reset_index()
        
        # Calculate metrics
        from imbalance_metrics import ImbalanceMetrics
        metrics_calc = ImbalanceMetrics()
        scorecard = metrics_calc.add_metrics_to_dataframe(scorecard)
        
        # Rank
        scorecard['priority_rank'] = scorecard['aer'].rank(ascending=True).astype(int)
        
        # Add status
        scorecard['status'] = scorecard['aer_grade'].map({
            'GREEN': '✓ Acceptable',
            'YELLOW': '⚠ Needs Improvement',
            'RED': '⚠ Critical'
        })
        
        # Sort by priority
        scorecard = scorecard.sort_values('priority_rank')
        
        print(f"Created scorecard for {len(scorecard)} districts")
        print(f"{'='*60}\n")
        
        return scorecard
    
    def print_rankings_summary(self):
        """Print formatted summary of rankings"""
        if self.district_rankings is None:
            print("No district rankings available. Run rank_districts() first.")
            return
        
        print(f"\n{'='*60}")
        print("DISTRICT RANKINGS SUMMARY")
        print(f"{'='*60}\n")
        
        for _, row in self.district_rankings.iterrows():
            print(f"{row['rank']}. {row['district']}")
            print(f"   AER: {row['aer']*100:.2f}% | AEBI: {row['aebi']:.1f} | Grade: {row['aer_grade']}")
            print(f"   Total Enrollments: {row['total_enrollments']:,}")
            print(f"   Priority: {row['priority']}")
            print()
        
        print(f"{'='*60}\n")


def rank_regions(df: pd.DataFrame, 
                 min_enrollments: int = 10,
                 print_summary: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to rank both districts and PIN codes
    
    Args:
        df: DataFrame with calculated metrics
        min_enrollments: Minimum enrollments for PIN code ranking
        print_summary: Whether to print summary
        
    Returns:
        Tuple of (district_rankings, pincode_rankings)
    """
    engine = RankingEngine()
    
    district_rankings = engine.rank_districts(df)
    pincode_rankings = engine.rank_pincodes(df, min_enrollments=min_enrollments)
    
    if print_summary:
        engine.print_rankings_summary()
    
    return district_rankings, pincode_rankings


if __name__ == "__main__":
    # Test ranking engine
    from data_loader import load_aadhaar_data
    from preprocessing import preprocess_aadhaar_data
    from imbalance_metrics import calculate_metrics
    
    df = load_aadhaar_data("../data/raw")
    df_clean = preprocess_aadhaar_data(df, aggregate_monthly=True)
    df_metrics = calculate_metrics(df_clean, print_summary=False)
    
    district_rankings, pincode_rankings = rank_regions(df_metrics)
    
    print("\nTop 10 Priority PIN Codes:")
    print(pincode_rankings[['district', 'pincode', 'aer_pct', 'risk_level', 
                            'recommendation']].head(10))