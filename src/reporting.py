"""
Reporting Module
Generates formatted reports, tables, and exports
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class ReportGenerator:
    """
    Generates analysis reports and visualizations
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory for saving outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'charts').mkdir(exist_ok=True)
        (self.output_dir / 'rankings').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def generate_executive_summary(self, df: pd.DataFrame) -> str:
        """
        Generate executive summary text
        
        Args:
            df: DataFrame with calculated metrics
            
        Returns:
            Formatted executive summary string
        """
        # Calculate aggregate statistics
        total_enrollments = df['total_enrollments'].sum()
        total_child = df['age_0_5'].sum()
        total_youth = df['age_5_17'].sum()
        total_adult = df['age_18_greater'].sum()
        
        avg_aer = df['aer'].mean()
        num_districts = df['district'].nunique()
        num_pincodes = df['pincode'].nunique()
        
        # Grade distribution
        grade_dist = df['aer_grade'].value_counts()
        
        summary = f"""
{'='*70}
EXECUTIVE SUMMARY - AADHAAR AGE-WISE ENROLLMENT ANALYSIS
{'='*70}

OVERVIEW
--------
Analysis Period: {df['year_month'].min()} to {df['year_month'].max()}
Geographic Scope: {num_districts} districts, {num_pincodes} PIN codes
Total Enrollments Analyzed: {total_enrollments:,}

AGE-WISE ENROLLMENT DISTRIBUTION
--------------------------------
Children (0-5 years):   {total_child:,} ({total_child/total_enrollments*100:.1f}%)
Youth (5-17 years):     {total_youth:,} ({total_youth/total_enrollments*100:.1f}%)
Adults (18+ years):     {total_adult:,} ({total_adult/total_enrollments*100:.1f}%)

KEY FINDINGS
------------
• Average Adult Enrollment Ratio (AER): {avg_aer*100:.2f}%
• Child enrollments dominate at {total_child/total_enrollments*100:.1f}% of total
• Adult enrollment severely limited at {total_adult/total_enrollments*100:.1f}%

PERFORMANCE GRADES
------------------
"""
        for grade in ['RED', 'YELLOW', 'GREEN']:
            count = grade_dist.get(grade, 0)
            pct = count / len(df) * 100 if len(df) > 0 else 0
            summary += f"{grade:7s}: {count:4d} regions ({pct:5.1f}%)\n"
        
        summary += f"""
CRITICAL INSIGHTS
-----------------
• For every 1 adult enrolled, approximately {int(total_child/max(total_adult, 1))} children are enrolled
• Geographic variation in adult enrollment is significant (range: {df['aer'].min()*100:.2f}% - {df['aer'].max()*100:.2f}%)
• Immediate intervention needed in {grade_dist.get('RED', 0)} high-priority zones

RECOMMENDED ACTIONS
-------------------
1. Deploy mobile enrollment units to RED-grade regions
2. Launch targeted adult awareness campaigns
3. Optimize enrollment processes at existing centers
4. Monitor progress monthly using AER as key metric

{'='*70}
"""
        return summary
    
    def export_rankings_to_excel(self, 
                                 district_rankings: pd.DataFrame,
                                 pincode_rankings: pd.DataFrame,
                                 filename: str = "enrollment_rankings.xlsx"):
        """
        Export rankings to Excel with multiple sheets
        
        Args:
            district_rankings: District ranking DataFrame
            pincode_rankings: PIN code ranking DataFrame
            filename: Output filename
        """
        filepath = self.output_dir / 'rankings' / filename
        
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            # District rankings
            district_rankings.to_excel(writer, sheet_name='District Rankings', index=False)
            
            # PIN code rankings
            pincode_rankings.to_excel(writer, sheet_name='PIN Code Rankings', index=False)
            
            # Priority zones (RED grade only)
            priority_zones = pincode_rankings[pincode_rankings['aer_grade'] == 'RED']
            priority_zones.to_excel(writer, sheet_name='Priority Zones', index=False)
            
            # Summary statistics
            summary_data = {
                'Metric': ['Total Districts', 'Total PIN Codes', 'Avg AER (%)', 
                          'RED Zones', 'YELLOW Zones', 'GREEN Zones'],
                'Value': [
                    len(district_rankings),
                    len(pincode_rankings),
                    f"{district_rankings['aer'].mean()*100:.2f}",
                    len(priority_zones),
                    len(pincode_rankings[pincode_rankings['aer_grade'] == 'YELLOW']),
                    len(pincode_rankings[pincode_rankings['aer_grade'] == 'GREEN'])
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Rankings exported to: {filepath}")
    
    def create_district_comparison_chart(self, 
                                         district_rankings: pd.DataFrame,
                                         save: bool = True) -> plt.Figure:
        """
        Create bar chart comparing districts
        
        Args:
            district_rankings: District ranking DataFrame
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Chart 1: AER comparison
        colors = district_rankings['aer_grade'].map({
            'RED': '#e74c3c',
            'YELLOW': '#f39c12',
            'GREEN': '#27ae60'
        })
        
        ax1.barh(district_rankings['district'], 
                district_rankings['aer_pct'],
                color=colors)
        ax1.set_xlabel('Adult Enrollment Ratio (%)')
        ax1.set_title('District Performance - Adult Enrollment Ratio')
        ax1.axvline(x=1, color='red', linestyle='--', label='Critical Threshold (1%)')
        ax1.axvline(x=5, color='green', linestyle='--', label='Target (5%)')
        ax1.legend()
        
        # Chart 2: Total enrollments with age breakdown
        districts = district_rankings['district']
        child = district_rankings['age_0_5']
        youth = district_rankings['age_5_17']
        adult = district_rankings['age_18_greater']
        
        ax2.barh(districts, child, label='0-5 years', color='#3498db')
        ax2.barh(districts, youth, left=child, label='5-17 years', color='#9b59b6')
        ax2.barh(districts, adult, left=child+youth, label='18+ years', color='#e74c3c')
        
        ax2.set_xlabel('Total Enrollments')
        ax2.set_title('District Enrollment Volume by Age Group')
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'charts' / 'district_comparison.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {filepath}")
        
        return fig
    
    def create_temporal_trend_chart(self, 
                                   df: pd.DataFrame,
                                   save: bool = True) -> plt.Figure:
        """
        Create line chart showing temporal trends
        
        Args:
            df: DataFrame with time series data
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Aggregate by month and district
        monthly = df.groupby(['year_month', 'district']).agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum',
            'total_enrollments': 'sum'
        }).reset_index()
        
        # Calculate AER
        from imbalance_metrics import ImbalanceMetrics
        metrics = ImbalanceMetrics()
        monthly = metrics.add_metrics_to_dataframe(monthly)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Chart 1: Total enrollments over time
        for district in monthly['district'].unique():
            district_data = monthly[monthly['district'] == district]
            ax1.plot(district_data['year_month'], 
                    district_data['total_enrollments'],
                    marker='o', label=district, linewidth=2)
        
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Total Enrollments')
        ax1.set_title('Monthly Enrollment Trends by District')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Chart 2: AER over time
        for district in monthly['district'].unique():
            district_data = monthly[monthly['district'] == district]
            ax2.plot(district_data['year_month'], 
                    district_data['aer_pct'],
                    marker='o', label=district, linewidth=2)
        
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Adult Enrollment Ratio (%)')
        ax2.set_title('Adult Enrollment Ratio Trends by District')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Critical (1%)')
        ax2.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='Target (5%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'charts' / 'temporal_trends.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {filepath}")
        
        return fig
    
    def create_pincode_heatmap(self, 
                              pincode_rankings: pd.DataFrame,
                              top_n: int = 30,
                              save: bool = True) -> plt.Figure:
        """
        Create heatmap of top PIN codes
        
        Args:
            pincode_rankings: PIN code ranking DataFrame
            top_n: Number of top PIN codes to show
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Select top N PIN codes
        top_pins = pincode_rankings.head(top_n).copy()
        
        # Prepare data for heatmap
        heatmap_data = top_pins[['pincode', 'child_pct', 'youth_pct', 'adult_pct']].set_index('pincode')
        heatmap_data.columns = ['0-5 years (%)', '5-17 years (%)', '18+ years (%)']
        
        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                   cbar_kws={'label': 'Percentage'}, ax=ax)
        
        ax.set_title(f'Age Distribution Heatmap - Top {top_n} Priority PIN Codes')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('PIN Code')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'charts' / 'pincode_heatmap.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {filepath}")
        
        return fig
    
    def generate_full_report(self, 
                           df: pd.DataFrame,
                           district_rankings: pd.DataFrame,
                           pincode_rankings: pd.DataFrame):
        """
        Generate complete analysis report with all outputs
        
        Args:
            df: Main DataFrame with all data
            district_rankings: District ranking DataFrame
            pincode_rankings: PIN code ranking DataFrame
        """
        print(f"\n{'='*70}")
        print("GENERATING FULL ANALYSIS REPORT")
        print(f"{'='*70}\n")
        
        # 1. Executive summary
        print("Generating executive summary...")
        summary = self.generate_executive_summary(df)
        summary_path = self.output_dir / 'reports' / 'executive_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(summary)
        
        # 2. Export rankings
        print("\nExporting rankings to Excel...")
        self.export_rankings_to_excel(district_rankings, pincode_rankings)
        
        # 3. Generate charts
        print("\nGenerating visualizations...")
        self.create_district_comparison_chart(district_rankings)
        self.create_temporal_trend_chart(df)
        self.create_pincode_heatmap(pincode_rankings)
        
        print(f"\n{'='*70}")
        print("REPORT GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nOutputs saved to: {self.output_dir.absolute()}")
        print(f"  - Executive summary: reports/executive_summary.txt")
        print(f"  - Rankings: rankings/enrollment_rankings.xlsx")
        print(f"  - Charts: charts/*.png")
        print(f"{'='*70}\n")


def generate_report(df: pd.DataFrame,
                   district_rankings: pd.DataFrame,
                   pincode_rankings: pd.DataFrame,
                   output_dir: str = "outputs"):
    """
    Convenience function to generate full report
    
    Args:
        df: Main DataFrame
        district_rankings: District rankings
        pincode_rankings: PIN code rankings
        output_dir: Output directory
    """
    generator = ReportGenerator(output_dir)
    generator.generate_full_report(df, district_rankings, pincode_rankings)


if __name__ == "__main__":
    # Test report generation
    from data_loader import load_aadhaar_data
    from preprocessing import preprocess_aadhaar_data
    from imbalance_metrics import calculate_metrics
    from ranking_engine import rank_regions
    
    df = load_aadhaar_data("../data/raw")
    df_clean = preprocess_aadhaar_data(df, aggregate_monthly=True)
    df_metrics = calculate_metrics(df_clean, print_summary=False)
    district_rankings, pincode_rankings = rank_regions(df_metrics, print_summary=False)
    
    generate_report(df_metrics, district_rankings, pincode_rankings)