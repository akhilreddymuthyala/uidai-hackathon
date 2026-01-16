"""
Streamlit Dashboard for Aadhaar Enrollment Analysis
Interactive web application for exploring enrollment imbalance data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from data_loader import load_aadhaar_data
from preprocessing import preprocess_aadhaar_data
from imbalance_metrics import calculate_metrics, ImbalanceMetrics
from ranking_engine import RankingEngine

# Page configuration
st.set_page_config(
    page_title="Aadhaar Enrollment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .red-grade {
        color: #e74c3c;
        font-weight: bold;
    }
    .yellow-grade {
        color: #f39c12;
        font-weight: bold;
    }
    .green-grade {
        color: #27ae60;
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_and_process_data(data_dir=None):
    """Load and process data with caching"""
    if data_dir is None:
        data_dir = ROOT_DIR / "data" / "raw"
    df = load_aadhaar_data(str(data_dir))
    df_clean = preprocess_aadhaar_data(df, aggregate_monthly=True)
    df_metrics = calculate_metrics(df_clean, print_summary=False)
    return df_metrics


@st.cache_data
def calculate_rankings(_df):
    """Calculate rankings with caching"""
    engine = RankingEngine()
    district_rankings = engine.rank_districts(_df)
    pincode_rankings = engine.rank_pincodes(_df, min_enrollments=10)
    priority_zones = engine.identify_priority_zones(_df)
    return district_rankings, pincode_rankings, priority_zones


def main():
    # Header
    st.markdown(
        '<p class="main-header">📊 Aadhaar Age-Wise Enrollment Analysis Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown("**UIDAI Data Hackathon 2026 | Telangana State Analysis**")
    st.markdown("---")

    # Load data
    with st.spinner("Loading and processing data..."):
        try:
            df = load_and_process_data()
            district_rankings, pincode_rankings, priority_zones = calculate_rankings(df)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.info("Please ensure data files are in the '../data/raw' directory")
            return

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # District filter
    districts = ["All"] + sorted(df["district"].unique().tolist())
    selected_district = st.sidebar.selectbox("Select District", districts)

    # Month filter
    months = ["All"] + sorted(df["year_month"].unique().tolist())
    selected_month = st.sidebar.selectbox("Select Month", months)

    # Grade filter
    grades = ["All", "RED", "YELLOW", "GREEN"]
    selected_grade = st.sidebar.selectbox("Select Performance Grade", grades)

    # Apply filters
    filtered_df = df.copy()
    if selected_district != "All":
        filtered_df = filtered_df[filtered_df["district"] == selected_district]
    if selected_month != "All":
        filtered_df = filtered_df[filtered_df["year_month"] == selected_month]
    if selected_grade != "All":
        filtered_df = filtered_df[filtered_df["aer_grade"] == selected_grade]

    # Key Metrics Row
    st.header("📈 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    total_enrollments = filtered_df["total_enrollments"].sum()
    total_adult = filtered_df["age_18_greater"].sum()
    avg_aer = filtered_df["aer"].mean() * 100
    red_zones = len(filtered_df[filtered_df["aer_grade"] == "RED"])
    num_districts = filtered_df["district"].nunique()

    with col1:
        st.metric("Total Enrollments", f"{total_enrollments:,}")
    with col2:
        st.metric("Adult Enrollments", f"{total_adult:,}")
    with col3:
        st.metric("Avg AER", f"{avg_aer:.2f}%")
    with col4:
        st.metric(
            "Critical Zones", red_zones, delta=f"{-red_zones}", delta_color="inverse"
        )
    with col5:
        st.metric("Districts Analyzed", num_districts)

    st.markdown("---")

    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Overview",
            "🗺️ District Analysis",
            "📍 PIN Code Analysis",
            "📈 Trends",
            "🎯 Priority Zones",
        ]
    )

    # Tab 1: Overview
    with tab1:
        st.header("Enrollment Distribution Overview")

        col1, col2 = st.columns(2)

        with col1:
            # Age distribution pie chart
            age_totals = {
                "0-5 years": filtered_df["age_0_5"].sum(),
                "5-17 years": filtered_df["age_5_17"].sum(),
                "18+ years": filtered_df["age_18_greater"].sum(),
            }

            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=list(age_totals.keys()),
                        values=list(age_totals.values()),
                        hole=0.4,
                        marker_colors=["#3498db", "#9b59b6", "#e74c3c"],
                    )
                ]
            )
            fig_pie.update_layout(title="Age Group Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Grade distribution
            grade_counts = filtered_df["aer_grade"].value_counts()

            grade_df = grade_counts.reset_index()
            grade_df.columns = ["aer_grade", "count"]

            fig_bar = px.bar(
                grade_df,
                x="aer_grade",
                y="count",
                color="aer_grade",
                labels={
                    "aer_grade": "Performance Grade",
                    "count": "Number of Regions",
                },
                title="Performance Grade Distribution",
                color_discrete_map={
                    "RED": "#e74c3c",
                    "YELLOW": "#f39c12",
                    "GREEN": "#27ae60",
                },
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)

        total_enrollments = (
            filtered_df["age_0_5"].sum()
            + filtered_df["age_5_17"].sum()
            + filtered_df["age_18_greater"].sum()
        )

        with col1:
            st.info(f"""
            **Children (0-5 years)**
            - Total: {filtered_df["age_0_5"].sum():,}
            - Percentage: {(filtered_df["age_0_5"].sum() / total_enrollments * 100) if total_enrollments > 0 else 0:.1f}%
            """)

        with col2:
            st.info(f"""
            **Youth (5-17 years)**
            - Total: {filtered_df["age_5_17"].sum():,}
            - Percentage: {(filtered_df["age_5_17"].sum() / total_enrollments * 100) if total_enrollments > 0 else 0:.1f}%
            """)

        with col3:
            st.info(f"""
            **Adults (18+ years)**
            - Total: {filtered_df["age_18_greater"].sum():,}
            - Percentage: {(filtered_df["age_18_greater"].sum() / total_enrollments * 100) if total_enrollments > 0 else 0:.1f}%
            """)


    # Tab 2: District Analysis
    with tab2:
        st.header("District Performance Comparison")

        # Filter district rankings
        display_rankings = district_rankings.copy()
        if selected_district != "All":
            display_rankings = display_rankings[
                display_rankings["district"] == selected_district
            ]

        # District comparison chart
        fig_district = px.bar(
            display_rankings,
            x="district",
            y="aer_pct",
            color="aer_grade",
            color_discrete_map={
                "RED": "#e74c3c",
                "YELLOW": "#f39c12",
                "GREEN": "#27ae60",
            },
            labels={"aer_pct": "Adult Enrollment Ratio (%)", "district": "District"},
            title="Adult Enrollment Ratio by District",
        )
        fig_district.add_hline(
            y=1, line_dash="dash", line_color="red", annotation_text="Critical (1%)"
        )
        fig_district.add_hline(
            y=5, line_dash="dash", line_color="green", annotation_text="Target (5%)"
        )
        st.plotly_chart(fig_district, use_container_width=True)

        # District ranking table
        st.subheader("District Rankings")
        st.dataframe(
            display_rankings[
                [
                    "rank",
                    "district",
                    "total_enrollments",
                    "aer_pct",
                    "aebi",
                    "aer_grade",
                    "priority",
                ]
            ].style.format(
                {"total_enrollments": "{:,.0f}", "aer_pct": "{:.2f}%", "aebi": "{:.1f}"}
            ),
            use_container_width=True,
        )

    # Tab 3: PIN Code Analysis
    with tab3:
        st.header("PIN Code Level Analysis")

        # Filter PIN code rankings
        display_pins = pincode_rankings.copy()
        if selected_district != "All":
            display_pins = display_pins[display_pins["district"] == selected_district]

        # Top and bottom performers
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏆 Top Performers")
            top_pins = display_pins.nlargest(10, "aer_pct")
            st.dataframe(
                top_pins[
                    ["pincode", "district", "aer_pct", "total_enrollments"]
                ].style.format({"aer_pct": "{:.2f}%", "total_enrollments": "{:,.0f}"}),
                use_container_width=True,
            )

        with col2:
            st.subheader("⚠️ Bottom Performers")
            bottom_pins = display_pins.nsmallest(10, "aer_pct")
            st.dataframe(
                bottom_pins[
                    ["pincode", "district", "aer_pct", "total_enrollments"]
                ].style.format({"aer_pct": "{:.2f}%", "total_enrollments": "{:,.0f}"}),
                use_container_width=True,
            )

        # Scatter plot
        st.subheader("PIN Code Performance Scatter")
        fig_scatter = px.scatter(
            display_pins,
            x="total_enrollments",
            y="aer_pct",
            color="risk_level",
            hover_data=["pincode", "district"],
            labels={"total_enrollments": "Total Enrollments", "aer_pct": "AER (%)"},
            title="PIN Code Performance: Enrollment Volume vs Adult Ratio",
            color_discrete_map={
                "HIGH": "#e74c3c",
                "MEDIUM": "#f39c12",
                "LOW": "#27ae60",
            },
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Tab 4: Trends
    with tab4:
        st.header("Temporal Trends Analysis")

        # Monthly trends
        monthly_data = (
            filtered_df.groupby("year_month")
            .agg(
                {
                    "total_enrollments": "sum",
                    "age_0_5": "sum",
                    "age_5_17": "sum",
                    "age_18_greater": "sum",
                }
            )
            .reset_index()
        )

        # Calculate AER for monthly data
        monthly_data["aer_pct"] = (
            monthly_data["age_18_greater"] / monthly_data["total_enrollments"] * 100
        )

        # Line charts
        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=monthly_data["year_month"],
                y=monthly_data["total_enrollments"],
                mode="lines+markers",
                name="Total Enrollments",
                line=dict(color="#3498db", width=3),
            )
        )
        fig_trend.update_layout(
            title="Monthly Enrollment Trends",
            xaxis_title="Month",
            yaxis_title="Total Enrollments",
            hovermode="x unified",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # AER trend
        fig_aer_trend = go.Figure()

        for district in filtered_df["district"].unique():
            district_monthly = (
                filtered_df[filtered_df["district"] == district]
                .groupby("year_month")
                .agg({"age_18_greater": "sum", "total_enrollments": "sum"})
                .reset_index()
            )
            district_monthly["aer_pct"] = (
                district_monthly["age_18_greater"]
                / district_monthly["total_enrollments"]
                * 100
            )

            fig_aer_trend.add_trace(
                go.Scatter(
                    x=district_monthly["year_month"],
                    y=district_monthly["aer_pct"],
                    mode="lines+markers",
                    name=district,
                )
            )

        fig_aer_trend.add_hline(
            y=1, line_dash="dash", line_color="red", annotation_text="Critical (1%)"
        )
        fig_aer_trend.add_hline(
            y=5, line_dash="dash", line_color="green", annotation_text="Target (5%)"
        )
        fig_aer_trend.update_layout(
            title="Adult Enrollment Ratio Trends by District",
            xaxis_title="Month",
            yaxis_title="AER (%)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_aer_trend, use_container_width=True)

    # Tab 5: Priority Zones
    with tab5:
        st.header("🎯 Priority Intervention Zones")

        st.info("""
        These zones require immediate attention due to critically low adult enrollment rates.
        Priority tiers are assigned based on AER thresholds.
        """)

        # Display priority zones
        if selected_district != "All":
            priority_display = priority_zones[
                priority_zones["district"] == selected_district
            ]
        else:
            priority_display = priority_zones

        # Tier breakdown
        tier_counts = priority_display["tier"].value_counts()
        col1, col2, col3 = st.columns(3)

        with col1:
            tier1_count = tier_counts.get("TIER 1 - URGENT", 0)
            st.metric(
                "TIER 1 - URGENT", tier1_count, delta=f"Critical", delta_color="inverse"
            )

        with col2:
            tier2_count = tier_counts.get("TIER 2 - MODERATE", 0)
            st.metric(
                "TIER 2 - MODERATE",
                tier2_count,
                delta="Needs attention",
                delta_color="off",
            )

        with col3:
            tier3_count = tier_counts.get("TIER 3 - LOW PRIORITY", 0)
            st.metric(
                "TIER 3 - LOW PRIORITY", tier3_count, delta="Monitor", delta_color="off"
            )


        # Priority zones table
        st.subheader("Detailed Priority Zones")
        # st.write("DEBUG priority_display columns:")
        # st.write(list(priority_display.columns))
        # st.stop()

        st.dataframe(
        priority_display[
                [
                    "tier",
                    "district",
                    "total_enrollments",
                    "aer_pct",
                ]
        ].style.format(
            {
                "total_enrollments": "{:,.0f}",
                "aer_pct": "{:.2f}%"
            }
        ),
        use_container_width=True,
        )



        # Download button
        csv = priority_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Priority Zones CSV",
            data=csv,
            file_name="priority_zones.csv",
            mime="text/csv",
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #666;'>
    <p>Aadhaar Age-Wise Enrollment Analysis Dashboard</p>
    <p>UIDAI Data Hackathon 2026 | Team Submission</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
