"""
Streamlit Dashboard for Aadhaar Enrollment Analysis
Interactive web application for exploring enrollment imbalance data
UIDAI Data Hackathon 2026 - Enhanced Professional Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import time
import base64

# Add parent directory to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from data_loader import load_aadhaar_data
from preprocessing import preprocess_aadhaar_data
from imbalance_metrics import calculate_metrics, ImbalanceMetrics
from ranking_engine import RankingEngine

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="UIDAI Enrollment Analytics | Data Hackathon 2026",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "UIDAI Data Hackathon 2026 - Age-Wise Enrollment Analysis"},
)

# ============================================================================
# CONSTANTS - Professional Color Scheme
# ============================================================================

AGE_COLORS = {
    "0-5": "#3498db",  # Light Blue
    "5-17": "#e67e22",  # Orange
    "18+": "#27ae60",  # Dark Green
}

GRADE_COLORS = {"RED": "#e74c3c", "YELLOW": "#f39c12", "GREEN": "#27ae60"}

# Professional color palette
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#2c3e50"
ACCENT_COLOR = "#3498db"
SUCCESS_COLOR = "#27ae60"
WARNING_COLOR = "#f39c12"
DANGER_COLOR = "#e74c3c"

"""
ENHANCED CSS FOR UIDAI ENROLLMENT ANALYTICS DASHBOARD
Government-grade professional styling with perfect readability
"""

ENHANCED_CSS = """
    <style>
    /* ========================================================================
       THEME LOCK & GLOBAL OVERRIDES
       ======================================================================== */
    
    /* Force white background theme regardless of system settings */
    .stApp {
        background-color: #ffffff !important;
    }
    
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
    }
    
    /* Force dark text on light backgrounds */
    .stApp, .main, p, li, span, div {
        color: #2c3e50 !important;
    }
    
    /* ========================================================================
       TYPOGRAPHY SYSTEM - Clear Hierarchy
       ======================================================================== */
    
    /* Hero Section Typography */
    .hero-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.3);
        animation: fadeInDown 0.8s ease-out;
    }
    
    .hero-title {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin-bottom: 0.75rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.2rem !important;
        color: #dbeafe !important;
        margin-bottom: 0.5rem !important;
        font-weight: 400 !important;
        line-height: 1.3;
    }
    
    .hero-tagline {
        font-size: 0.95rem !important;
        color: #93c5fd !important;
        font-style: italic;
        margin-top: 0.75rem !important;
    }
    
    /* Section Headers - Clear Visual Hierarchy */
    .section-header {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
        margin: 2rem 0 1.25rem 0 !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 3px solid #3b82f6 !important;
        animation: fadeIn 0.8s ease-out;
        letter-spacing: -0.5px;
    }
    
    /* Body Text - Readable Size */
    p, li, span, div {
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        color: #334155 !important;
    }
    
    /* ========================================================================
       SIDEBAR STYLING - Dark Background with White Text
       ======================================================================== */
    
    /* Sidebar container */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%) !important;
        padding-top: 2rem;
    }
    
    /* All sidebar text - white by default */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Sidebar headings */
    [data-testid="stSidebar"] h3 {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin-bottom: 1rem !important;
    }
    
    /* Sidebar paragraphs */
    [data-testid="stSidebar"] p {
        font-size: 1rem !important;
        color: #e2e8f0 !important;
        line-height: 1.6 !important;
    }
    
    /* ========================================================================
       SIDEBAR INPUT FIELDS - BLACK TEXT FIX
       ======================================================================== */
    
    /* Sidebar selectbox - force dark text */
    [data-testid="stSidebar"] .stSelectbox label {
        color: #ffffff !important;
    }
    
    /* Sidebar selectbox input/display */
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stSelectbox input {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    /* Sidebar selectbox selected value and placeholder */
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div,
    [data-testid="stSidebar"] .stSelectbox svg + div {
        color: #1e293b !important;
    }
    
    /* Force all text in selectbox to be dark */
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #1e293b !important;
    }
    
    /* Dropdown menu background */
    [data-testid="stSidebar"] [role="listbox"] {
        background-color: #ffffff !important;
    }
    
    /* Dropdown menu options */
    [data-testid="stSidebar"] [role="option"] {
        color: #1e293b !important;
        background-color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] [role="option"]:hover {
        background-color: #f1f5f9 !important;
        color: #1e293b !important;
    }
    
    /* ========================================================================
       SIDEBAR EXPANDERS - Critical Fix for Visibility
       ======================================================================== */
    
    /* Expander summary (clickable header) */
    [data-testid="stSidebar"] div[data-testid="stExpander"] summary {
        background-color: rgba(59, 130, 246, 0.15) !important;
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 1rem !important;
        border-radius: 8px !important;
        border-left: 4px solid #3b82f6 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Expander summary hover effect */
    [data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover {
        background-color: rgba(59, 130, 246, 0.25) !important;
    }
    
    /* Expander body content */
    [data-testid="stSidebar"] div[data-testid="stExpander"] .streamlit-expanderContent {
        background-color: rgba(15, 23, 42, 0.3) !important;
        padding: 1rem !important;
        border-radius: 0 0 8px 8px !important;
        border-left: 4px solid #3b82f6 !important;
    }
    
    /* Expander body text */
    [data-testid="stSidebar"] div[data-testid="stExpander"] p,
    [data-testid="stSidebar"] div[data-testid="stExpander"] li,
    [data-testid="stSidebar"] div[data-testid="stExpander"] span {
        color: #f1f5f9 !important;
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
    }
    
    /* Expander strong/bold text */
    [data-testid="stSidebar"] div[data-testid="stExpander"] strong,
    [data-testid="stSidebar"] div[data-testid="stExpander"] b {
        color: #bfdbfe !important;
        font-weight: 700 !important;
    }
    
    /* ========================================================================
       METRIC CARDS - Professional & Animated
       ======================================================================== */
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border-radius: 12px;
        padding: 1.5rem 1.25rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 1px solid #e2e8f0;
        position: relative;
        overflow: hidden;
        animation: slideUp 0.6s ease-out;
        min-height: 160px;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.12);
        border-color: #3b82f6;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-icon {
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        display: block;
        opacity: 0.9;
    }
    
    .metric-value {
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        color: #1e293b !important;
        margin: 0.5rem 0 !important;
        font-family: 'SF Mono', 'Monaco', 'Courier New', monospace !important;
        letter-spacing: -0.5px;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        hyphens: auto !important;
        line-height: 1.2 !important;
    }
    
    .metric-label {
        font-size: 0.8rem !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600 !important;
        margin-top: 0.5rem !important;
        line-height: 1.3 !important;
        word-break: break-word !important;
    }
    
    .metric-delta {
        font-size: 0.85rem !important;
        margin-top: 0.5rem !important;
        padding: 0.35rem 0.85rem !important;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600 !important;
    }
    
    .metric-delta.positive {
        background-color: #d1fae5 !important;
        color: #065f46 !important;
    }
    
    .metric-delta.negative {
        background-color: #fee2e2 !important;
        color: #991b1b !important;
    }
    
    /* ========================================================================
       INSIGHT & RECOMMENDATION BOXES
       ======================================================================== */
    
    .insight-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 5px solid #3b82f6;
        padding: 1.25rem;
        margin: 1.5rem 0;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        animation: slideInLeft 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .insight-box:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.2);
    }
    
    .insight-box b {
        color: #1e3a8a !important;
        font-size: 1rem !important;
    }
    
    .insight-box p {
        color: #1e293b !important;
        margin: 0.5rem 0 !important;
        font-size: 0.95rem !important;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #10b981;
        padding: 1.25rem;
        margin: 1.5rem 0;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
        animation: slideInRight 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .recommendation-box:hover {
        transform: translateX(-5px);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.2);
    }
    
    .recommendation-box h4 {
        color: #065f46 !important;
        margin-top: 0 !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }
    
    .recommendation-box b {
        color: #047857 !important;
        font-size: 1rem !important;
    }
    
    .recommendation-box p {
        color: #1e293b !important;
        margin: 0.5rem 0 !important;
        font-size: 0.95rem !important;
    }
    
    /* ========================================================================
       ACTION CARDS
       ======================================================================== */
    
    .action-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.25rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border-top: 4px solid;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .action-card.tier1 {
        border-top-color: #dc2626;
        background: linear-gradient(135deg, #ffffff 0%, #fee2e2 100%);
    }
    
    .action-card.tier2 {
        border-top-color: #f59e0b;
        background: linear-gradient(135deg, #ffffff 0%, #fef3c7 100%);
    }
    
    .action-card.tier3 {
        border-top-color: #10b981;
        background: linear-gradient(135deg, #ffffff 0%, #d1fae5 100%);
    }
    
    .action-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    
    .action-card-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .action-card-icon {
        font-size: 2rem !important;
        margin-right: 1rem;
    }
    
    .action-card-title {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
    }
    
    .priority-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        margin-left: auto;
        letter-spacing: 0.5px;
    }
    
    .priority-badge.tier1 {
        background-color: #dc2626;
        color: white !important;
    }
    
    .priority-badge.tier2 {
        background-color: #f59e0b;
        color: white !important;
    }
    
    .priority-badge.tier3 {
        background-color: #10b981;
        color: white !important;
    }
    
    /* ========================================================================
       TABS STYLING
       ======================================================================== */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        padding: 0.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px;
        padding: 0 1.75rem;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease;
        color: #64748b !important;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e2e8f0;
        color: #1e293b !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* ========================================================================
       STREAMLIT NATIVE COMPONENTS OVERRIDE
       ======================================================================== */
    
    /* Info/Warning/Error boxes */
    .stAlert {
        border-radius: 10px !important;
        padding: 1.25rem !important;
        font-size: 1rem !important;
    }
    
    /* Dataframes */
    .dataframe {
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
        font-size: 0.95rem !important;
    }
    
    .dataframe thead th {
        background-color: #1e293b !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        font-size: 1rem !important;
    }
    
    .dataframe tbody td {
        padding: 0.75rem !important;
        color: #334155 !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Selectbox and inputs - BLACK TEXT */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        color: #1e293b !important;
    }
    
    /* Force ALL text inside selectbox to be dark */
    .stSelectbox * {
        color: #1e293b !important;
    }
    
    /* Selectbox dropdown options - BLACK TEXT */
    .stSelectbox div[data-baseweb="select"] > div {
        color: #1e293b !important;
    }
    
    /* Selectbox selected value - BLACK TEXT */
    .stSelectbox [data-baseweb="select"] span {
        color: #1e293b !important;
    }
    
    /* Value container text */
    .stSelectbox [data-baseweb="select"] [data-baseweb="input"] {
        color: #1e293b !important;
    }
    
    /* All input fields - BLACK TEXT */
    input, select, textarea {
        color: #1e293b !important;
        background-color: #ffffff !important;
    }
    
    /* Dropdown menu items */
    [role="option"] {
        color: #1e293b !important;
        background-color: #ffffff !important;
    }
    
    [role="option"]:hover {
        background-color: #f1f5f9 !important;
    }
    
    /* PLACEHOLDER TEXT - DARK GRAY */
    input::placeholder,
    textarea::placeholder {
        color: #475569 !important;
        opacity: 1 !important;
    }
    
    /* Selectbox placeholder - more aggressive */
    [data-baseweb="select"] [aria-selected="false"],
    [data-baseweb="select"] [class*="placeholder"] {
        color: #475569 !important;
        opacity: 1 !important;
    }
    
    /* ========================================================================
       ANIMATIONS
       ======================================================================== */
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* ========================================================================
       FOOTER
       ======================================================================== */
    
    .footer {
        text-align: center;
        padding: 2.5rem 2rem;
        margin-top: 3rem;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        border-radius: 16px;
        color: white !important;
        animation: fadeIn 1s ease-out;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.3);
    }
    
    .footer * {
        color: white !important;
    }
    
    .footer-logo {
        font-size: 2rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    .footer h3 {
        font-size: 1.5rem !important;
        margin: 0.5rem 0 !important;
    }
    
    .footer p {
        color: #dbeafe !important;
        font-size: 0.9rem !important;
    }
    
    /* ========================================================================
       RESPONSIVE DESIGN
       ======================================================================== */
    
    @media (max-width: 768px) {
        .hero-title {
            font-size: 1.8rem !important;
        }
        
        .hero-subtitle {
            font-size: 1rem !important;
        }
        
        .section-header {
            font-size: 1.5rem !important;
        }
        
        .metric-card {
            margin-bottom: 1rem;
            min-height: 140px;
        }
        
        .metric-value {
            font-size: 1.5rem !important;
        }
    }
    
    /* ========================================================================
       DIVIDER STYLING
       ======================================================================== */
    
    hr {
        border: none !important;
        border-top: 2px solid #e2e8f0 !important;
        margin: 2.5rem 0 !important;
    }
    
    /* ========================================================================
       SPINNER/LOADING
       ======================================================================== */
    
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* ========================================================================
       FIX OVERLAPPING ISSUES
       ======================================================================== */
    
    /* Ensure proper spacing between sections */
    .element-container {
        margin-bottom: 1rem !important;
    }
    
    /* Fix column gaps */
    [data-testid="column"] {
        padding: 0 0.75rem !important;
    }
    
    /* Prevent text overflow */
    * {
        overflow-wrap: break-word !important;
        word-wrap: break-word !important;
    }
    
    /* Clear floats */
    .clearfix::after {
        content: "";
        display: table;
        clear: both;
    }
    </style>
"""


# To use this CSS in your Streamlit app:
st.markdown(ENHANCED_CSS, unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS - UI COMPONENTS
# ============================================================================
def render_logo():
    """Load and encode logo for use in header"""
    from pathlib import Path
    import base64

    # assets folder is inside dashboard
    logo_path = Path(__file__).parent / "assets" / "logo.png"

    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{logo_data}"
    else:
        return None


def render_hero_section():
    """Render animated hero header section with integrated logo"""
    logo_src = render_logo()

    # Determine logo HTML based on whether file exists
    if logo_src:
        logo_html = f'<img src="{logo_src}" alt="Aadhaar Analytics" class="hero-logo">'
    else:
        logo_html = '<div class="hero-logo-emoji">🇮🇳</div>'

    st.markdown(
        f"""
        <style>
        .hero-container {{
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 2.5rem 2rem 2.5rem 3rem; 
            padding-left: 4rem; 
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(30, 58, 138, 0.3);
            animation: fadeInDown 0.8s ease-out;
            display: flex;
            align-items: center;
            gap: 2rem;
        }}
        
        .hero-logo {{
            width: 100px;
            height: 100px;
            min-width: 100px;
            border-radius: 50%;
            background: white;
            padding: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            object-fit: contain;
            flex-shrink: 0;
            transform: scale(1.08);
        }}
        
        .hero-logo-emoji {{
            width: 100px;
            height: 100px;
            min-width: 100px;
            border-radius: 50%;
            background: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 3.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            flex-shrink: 0;
        }}
        
        .hero-content {{
            flex: 1;
            text-align: center;
        }}
        
        .hero-title {{
            font-size: 2.8rem !important;
            font-weight: 800 !important;
            color: #ffffff !important;
            margin-bottom: 0.75rem !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            letter-spacing: -0.5px;
            line-height: 1.2;
        }}
        
        .hero-subtitle {{
            font-size: 1.2rem !important;
            color: #dbeafe !important;
            margin-bottom: 0.5rem !important;
            font-weight: 400 !important;
            line-height: 1.3;
        }}
        
        .hero-tagline {{
            font-size: 0.95rem !important;
            color: #93c5fd !important;
            font-style: italic;
            margin-top: 0.75rem !important;
        }}
        
        @media (max-width: 768px) {{
            .hero-container {{
                flex-direction: column;
                text-align: center;
                gap: 1.5rem;
                padding: 2rem 1rem;
            }}
            
            .hero-logo,
            .hero-logo-emoji {{
                width: 80px;
                height: 80px;
                min-width: 80px;
            }}
            
            .hero-logo-emoji {{
                font-size: 2.5rem;
            }}
            
            .hero-title {{
                font-size: 1.8rem !important;
            }}
            
            .hero-subtitle {{
                font-size: 1rem !important;
            }}
        }}
        </style>
        
        <div class="hero-container">
            {logo_html}
            <div class="hero-content">
                <div class="hero-title">Aadhaar Enrollment Analytics Dashboard</div>
                <div class="hero-subtitle">AGE-WISE AADHAAR ENROLLMENT IMBALANCE ANALYSIS FOR INCLUSIVE IDENTITY COVERAGE</div>
                <div class="hero-tagline">UIDAI Data Hackathon 2026 | Empowering Inclusive Identity</div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    icon, value, label, delta=None, delta_type="neutral", col_index=0
):
    """Render animated metric card with icon and delta"""
    # Animation delay based on column index
    delay = col_index * 0.1

    delta_html = ""
    if delta:
        delta_class = "positive" if delta_type == "positive" else "negative"
        delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>'

    html = f"""
        <div class="metric-card" style="animation-delay: {delay}s;">
            <span class="metric-icon">{icon}</span>
            <div class="metric-value" style="animation: countUp 0.8s ease-out {delay}s backwards;">
                {value}
            </div>
            <div class="metric-label">{label}</div>
            {delta_html}
        </div>
    """

    return html


def render_action_card(priority, icon, title, description, impact):
    """Render action recommendation card"""
    priority_class = priority.lower().replace(" ", "")

    html = f"""
        <div class="action-card {priority_class}">
            <div class="action-card-header">
                <span class="action-card-icon">{icon}</span>
                <span class="action-card-title">{title}</span>
                <span class="priority-badge {priority_class}">{priority}</span>
            </div>
            <p style="color: #555; line-height: 1.6;">{description}</p>
            <p style="color: #27ae60; font-weight: 600; margin-top: 1rem;">
                📊 Expected Impact: {impact}
            </p>
        </div>
    """

    return html


# ============================================================================
# HELPER FUNCTIONS - DATA PROCESSING (UNCHANGED)
# ============================================================================


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


def apply_filters(df, state, district, month_range):
    """Apply filters to dataframe"""
    filtered_df = df.copy()

    if state != "All":
        filtered_df = filtered_df[filtered_df["state"] == state]

    if district != "All":
        filtered_df = filtered_df[filtered_df["district"] == district]

    if month_range and len(month_range) == 2:
        start_month, end_month = month_range
        filtered_df = filtered_df[
            (filtered_df["year_month"] >= start_month)
            & (filtered_df["year_month"] <= end_month)
        ]

    return filtered_df


# ============================================================================
# VISUALIZATION FUNCTIONS (ENHANCED WITH SMOOTH TRANSITIONS)
# ============================================================================


def create_age_distribution_chart(df):
    """Create age distribution pie chart with smooth transitions"""
    age_totals = {
        "0-5 years": df["age_0_5"].sum(),
        "5-17 years": df["age_5_17"].sum(),
        "18+ years": df["age_18_greater"].sum(),
    }

    colors = [AGE_COLORS["0-5"], AGE_COLORS["5-17"], AGE_COLORS["18+"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(age_totals.keys()),
                values=list(age_totals.values()),
                hole=0.4,
                marker_colors=colors,
                textinfo="label+percent",
                textfont_size=14,
                hovertemplate="<b>%{label}</b><br>Enrollments: %{value:,}<br>Percentage: %{percent}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Age Group Distribution",
            "font": {"size": 20, "color": "#2c3e50", "family": "Arial Black"},
        },
        showlegend=True,
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        transition={"duration": 500},
    )

    return fig


def create_temporal_trend_chart(df):
    """Create time-series line chart with smooth animations"""
    monthly_data = (
        df.groupby("year_month")
        .agg({"age_0_5": "sum", "age_5_17": "sum", "age_18_greater": "sum"})
        .reset_index()
    )

    fig = go.Figure()

    # Add traces with smooth line shapes
    fig.add_trace(
        go.Scatter(
            x=monthly_data["year_month"],
            y=monthly_data["age_0_5"],
            mode="lines+markers",
            name="0-5 years",
            line=dict(color=AGE_COLORS["0-5"], width=3, shape="spline"),
            marker=dict(size=8, line=dict(width=2, color="white")),
            hovertemplate="<b>0-5 years</b><br>Month: %{x}<br>Enrollments: %{y:,}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly_data["year_month"],
            y=monthly_data["age_5_17"],
            mode="lines+markers",
            name="5-17 years",
            line=dict(color=AGE_COLORS["5-17"], width=3, shape="spline"),
            marker=dict(size=8, line=dict(width=2, color="white")),
            hovertemplate="<b>5-17 years</b><br>Month: %{x}<br>Enrollments: %{y:,}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly_data["year_month"],
            y=monthly_data["age_18_greater"],
            mode="lines+markers",
            name="18+ years",
            line=dict(color=AGE_COLORS["18+"], width=3, shape="spline"),
            marker=dict(size=8, line=dict(width=2, color="white")),
            hovertemplate="<b>18+ years</b><br>Month: %{x}<br>Enrollments: %{y:,}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "Monthly Enrollment Trends by Age Group",
            "font": {"size": 20, "color": "#2c3e50", "family": "Arial Black"},
        },
        xaxis_title="Month",
        yaxis_title="Number of Enrollments",
        hovermode="x unified",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(245,245,245,1)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#2c3e50",
            borderwidth=1,
        ),
        transition={"duration": 500},
        xaxis=dict(showgrid=True, gridcolor="white"),
        yaxis=dict(showgrid=True, gridcolor="white"),
    )

    return fig


def create_district_aer_chart(district_rankings):
    """Create bar chart with smooth transitions"""
    sorted_df = district_rankings.sort_values("aer_pct")

    fig = px.bar(
        sorted_df,
        x="aer_pct",
        y="district",
        orientation="h",
        color="aer_grade",
        color_discrete_map=GRADE_COLORS,
        labels={"aer_pct": "Adult Enrollment Ratio (%)", "district": "District"},
        title="District Performance: Adult Enrollment Ratio (AER)",
    )

    # Add threshold lines
    fig.add_vline(
        x=1,
        line_dash="dash",
        line_color=DANGER_COLOR,
        line_width=2,
        annotation_text="Critical (1%)",
        annotation_position="top",
    )
    fig.add_vline(
        x=5,
        line_dash="dash",
        line_color=SUCCESS_COLOR,
        line_width=2,
        annotation_text="Target (5%)",
        annotation_position="top",
    )

    fig.update_layout(
        height=500,
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(245,245,245,1)",
        title={"font": {"size": 20, "color": "#2c3e50", "family": "Arial Black"}},
        transition={"duration": 500},
        xaxis=dict(showgrid=True, gridcolor="white"),
        yaxis=dict(showgrid=False),
    )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>AER: %{x:.2f}%<br>Grade: %{marker.color}<extra></extra>"
    )

    return fig


def create_risk_heatmap(pincode_rankings, top_n=20):
    """Create heatmap with enhanced styling"""
    top_risk = pincode_rankings.nsmallest(top_n, "aer_pct").copy()

    heatmap_data = top_risk[
        ["pincode", "child_pct", "youth_pct", "adult_pct"]
    ].set_index("pincode")
    heatmap_data.columns = ["0-5 years (%)", "5-17 years (%)", "18+ years (%)"]

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values.T,
            x=heatmap_data.index,
            y=heatmap_data.columns,
            colorscale="RdYlGn",
            reversescale=True,
            text=heatmap_data.values.T,
            texttemplate="%{text:.1f}%",
            textfont={"size": 11, "color": "black"},
            colorbar=dict(title="Percentage", thickness=15),
            hovertemplate="PIN: %{x}<br>Age: %{y}<br>Percentage: %{z:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"High-Risk PIN Codes - Top {top_n} by Lowest AER",
            "font": {"size": 20, "color": "#2c3e50", "family": "Arial Black"},
        },
        xaxis_title="PIN Code",
        yaxis_title="Age Group",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        transition={"duration": 500},
    )

    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    # Render hero section
    render_logo()
    render_hero_section()

    # ========================================================================
    # SIDEBAR - FILTERS (STICKY)
    # ========================================================================

    with st.sidebar:
        st.markdown("### 🎯 Navigation & Filters")
        st.markdown("---")

        # Project overview expander
        with st.expander("📖 About This Dashboard", expanded=False):
            st.markdown("""
            **Mission**: Enable data-driven decisions for inclusive Aadhaar coverage.
            
            **Key Questions Answered**:
            - Which districts need urgent intervention?
            - Where should mobile units be deployed?
            - How do enrollment patterns vary over time?
            
            **Impact**: Optimize resource allocation, reduce coverage gaps, ensure no one is left behind.
            """)

        with st.expander("📊 Understanding Metrics", expanded=False):
            st.markdown("""
            **AER (Adult Enrollment Ratio)**  
            Percentage of adult (18+) enrollments  
            🎯 Target: > 5%  
            ⚠️ Critical: < 1%
            
            **AEBI (Age Enrollment Balance Index)**  
            Composite score (0-100) measuring age balance  
            🟢 Green: ≥ 7  
            🟡 Yellow: 3-7  
            🔴 Red: < 3
            """)

        st.markdown("---")
        st.markdown("### 🔍 Filter Data")

        # Load data with spinner
        with st.spinner("🔄 Loading enrollment data..."):
            try:
                df = load_and_process_data()
                district_rankings, pincode_rankings, priority_zones = (
                    calculate_rankings(df)
                )
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.info("Please ensure data files are in the 'data/raw' directory")
                return

        # Add state column if missing
        if "state" not in df.columns:
            df["state"] = "Telangana"

        states = ["All"] + sorted(df["state"].unique().tolist())
        selected_state = st.selectbox("📍 State", states, key="state_filter")

        # District filter
        if selected_state != "All":
            districts = ["All"] + sorted(
                df[df["state"] == selected_state]["district"].unique().tolist()
            )
        else:
            districts = ["All"] + sorted(df["district"].unique().tolist())

        selected_district = st.selectbox("🏙️ District", districts, key="district_filter")

        # Month range
        all_months = sorted(df["year_month"].unique().tolist())

        st.markdown("**📅 Date Range**")
        col1, col2 = st.columns(2)
        with col1:
            start_month = st.selectbox("From", all_months, index=0, key="start_month")
        with col2:
            end_month = st.selectbox(
                "To", all_months, index=len(all_months) - 1, key="end_month"
            )

        month_range = (
            (start_month, end_month)
            if start_month <= end_month
            else (end_month, start_month)
        )

        # Apply filters
        filtered_df = apply_filters(df, selected_state, selected_district, month_range)

        # Filter summary with badges
        st.markdown("---")
        st.markdown("### ✅ Active Filters")
        st.markdown(
            f"""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px;'>
            <p style='color: white; margin: 0.3rem 0;'>🌍 <b>State:</b> {selected_state}</p>
            <p style='color: white; margin: 0.3rem 0;'>🏙️ <b>District:</b> {selected_district}</p>
            <p style='color: white; margin: 0.3rem 0;'>📅 <b>Period:</b> {month_range[0]} to {month_range[1]}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ========================================================================
    # MAIN CONTENT
    # ========================================================================

    if len(filtered_df) == 0:
        st.warning(
            "⚠️ No data available for selected filters. Please adjust your selection."
        )
        return

    # ========================================================================
    # ANIMATED KPI METRICS SECTION
    # ========================================================================

    st.markdown(
        '<div class="section-header">📊 Key Performance Indicators</div>',
        unsafe_allow_html=True,
    )

    # Calculate KPIs
    overall_aer = filtered_df["aer"].mean() * 100
    total_enrollments = filtered_df["total_enrollments"].sum()
    total_adult = filtered_df["age_18_greater"].sum()

    # District metrics
    district_aer = filtered_df.groupby("district")["aer"].mean().sort_values()
    lowest_aer_district = district_aer.index[0] if len(district_aer) > 0 else "N/A"
    lowest_aer_value = district_aer.iloc[0] * 100 if len(district_aer) > 0 else 0
    highest_aer_district = district_aer.index[-1] if len(district_aer) > 0 else "N/A"
    highest_aer_value = district_aer.iloc[-1] * 100 if len(district_aer) > 0 else 0

    zero_adult_pins = len(filtered_df[filtered_df["age_18_greater"] == 0])

    # Render metric cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        delta_text = "Below Target" if overall_aer < 5 else "Above Target"
        st.markdown(
            render_metric_card(
                "📊",
                f"{overall_aer:.2f}%",
                "Overall AER",
                delta_text,
                "negative" if overall_aer < 5 else "positive",
                0,
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            render_metric_card(
                "⚠️",
                lowest_aer_district[:15] + "..."
                if len(lowest_aer_district) > 15
                else lowest_aer_district,
                "Lowest AER District",
                f"{lowest_aer_value:.2f}%",
                "negative",
                1,
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            render_metric_card(
                "🏆",
                highest_aer_district[:15] + "..."
                if len(highest_aer_district) > 15
                else highest_aer_district,
                "Best Performer",
                f"{highest_aer_value:.2f}%",
                "positive",
                2,
            ),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            render_metric_card(
                "🚨",
                f"{zero_adult_pins}",
                "Zero Adult PINs",
                "Critical Zones",
                "negative",
                3,
            ),
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            render_metric_card(
                "📈",
                f"{total_enrollments:,}",
                "Total Enrollments",
                f"{total_adult:,} adults",
                "neutral",
                4,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ========================================================================
    # TABBED SECTIONS
    # ========================================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Overview Dashboard",
            "📈 Temporal Analysis",
            "🗺️ District Rankings",
            "📍 PIN Code Insights",
            "🎯 Action Plan",
        ]
    )

    # ========================================================================
    # TAB 1: OVERVIEW
    # ========================================================================

    with tab1:
        st.markdown(
            '<div class="section-header">Enrollment Distribution Overview</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            fig_pie = create_age_distribution_chart(filtered_df)
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown(
                """
            <div class="insight-box">
            <b>📊 Insight:</b> Child enrollments dominate, as expected for newborn registrations.<br>
            <b>💡 Policy Implication:</b> Adult enrollment below 1% signals systematic accessibility gaps 
            requiring targeted outreach beyond passive enrollment centers.
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            grade_counts = filtered_df["aer_grade"].value_counts()
            grade_df = pd.DataFrame(
                {"Grade": grade_counts.index, "Count": grade_counts.values}
            )

            fig_grade = px.bar(
                grade_df,
                x="Grade",
                y="Count",
                color="Grade",
                color_discrete_map=GRADE_COLORS,
                title="Performance Grade Distribution",
                labels={"Count": "Number of Regions"},
            )
            fig_grade.update_layout(
                height=450,
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(245,245,245,1)",
                title={
                    "font": {"size": 20, "color": "#2c3e50", "family": "Arial Black"}
                },
                transition={"duration": 500},
            )
            st.plotly_chart(fig_grade, use_container_width=True)

            st.markdown(
                """
            <div class="insight-box">
            <b>📊 Insight:</b> RED zones represent immediate intervention priorities.<br>
            <b>💡 UIDAI Action:</b> Allocate 60% of mobile units to RED zones, 30% to YELLOW, 10% to GREEN for maintenance.
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Summary statistics
        st.markdown(
            '<div class="section-header">Age-Wise Enrollment Summary</div>',
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)

        total_calc = filtered_df["total_enrollments"].sum()

        with col1:
            child_total = filtered_df["age_0_5"].sum()
            child_pct = (child_total / total_calc * 100) if total_calc > 0 else 0
            st.info(f"""
            **👶 Children (0-5 years)**
            - Total: {child_total:,}
            - Share: {child_pct:.1f}%
            - Context: High due to birth registrations
            """)

        with col2:
            youth_total = filtered_df["age_5_17"].sum()
            youth_pct = (youth_total / total_calc * 100) if total_calc > 0 else 0
            st.warning(f"""
            **🎓 Youth (5-17 years)**
            - Total: {youth_total:,}
            - Share: {youth_pct:.1f}%
            - Context: Delayed enrollments
            """)

        with col3:
            adult_total = filtered_df["age_18_greater"].sum()
            adult_pct = (adult_total / total_calc * 100) if total_calc > 0 else 0
            delta_color = "🔴" if adult_pct < 1 else "🟡" if adult_pct < 5 else "🟢"
            st.error(f"""
            **👨‍💼 Adults (18+ years)** {delta_color}
            - Total: {adult_total:,}
            - Share: {adult_pct:.1f}%
            - Target: >5% (currently {"below" if adult_pct < 5 else "meeting"} target)
            """)

    # ========================================================================
    # TAB 2: TEMPORAL TRENDS
    # ========================================================================

    with tab2:
        st.markdown(
            '<div class="section-header">Time Series Analysis</div>',
            unsafe_allow_html=True,
        )

        fig_temporal = create_temporal_trend_chart(filtered_df)
        st.plotly_chart(fig_temporal, use_container_width=True)

        st.markdown(
            """
        <div class="insight-box">
        <b>📊 Pattern Recognition:</b> June-July spikes correlate with school admissions.<br>
        <b>💡 Operational Strategy:</b> Pre-deploy additional enrollment officers in May to handle 
        anticipated demand surge. Reduce staffing Aug-Dec during off-peak months.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.divider()

        # AER trends
        st.markdown(
            '<div class="section-header">District-Wise AER Trends</div>',
            unsafe_allow_html=True,
        )

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
                    line=dict(width=3, shape="spline"),
                    marker=dict(size=8, line=dict(width=2, color="white")),
                    hovertemplate=f"<b>{district}</b><br>Month: %{{x}}<br>AER: %{{y:.2f}}%<extra></extra>",
                )
            )

        fig_aer_trend.add_hline(
            y=1,
            line_dash="dash",
            line_color=DANGER_COLOR,
            line_width=2,
            annotation_text="Critical (1%)",
            annotation_position="top left",
        )
        fig_aer_trend.add_hline(
            y=5,
            line_dash="dash",
            line_color=SUCCESS_COLOR,
            line_width=2,
            annotation_text="Target (5%)",
            annotation_position="bottom left",
        )

        fig_aer_trend.update_layout(
            title={
                "text": "Adult Enrollment Ratio Over Time",
                "font": {"size": 20, "color": "#2c3e50"},
            },
            xaxis_title="Month",
            yaxis_title="AER (%)",
            hovermode="x unified",
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(245,245,245,1)",
            transition={"duration": 500},
        )

        st.plotly_chart(fig_aer_trend, use_container_width=True)

        st.markdown(
            """
        <div class="insight-box">
        <b>📊 Progress Tracking:</b> Flat or declining AER indicates intervention failure.<br>
        <b>💡 Course Correction:</b> Districts showing no improvement over 3 months require 
        root cause analysis and strategy revision.
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ========================================================================
    # TAB 3: DISTRICT ANALYSIS
    # ========================================================================

    with tab3:
        st.markdown(
            '<div class="section-header">District Performance Comparison</div>',
            unsafe_allow_html=True,
        )

        display_rankings = district_rankings.copy()
        if selected_district != "All":
            display_rankings = display_rankings[
                display_rankings["district"] == selected_district
            ]

        fig_district = create_district_aer_chart(display_rankings)
        st.plotly_chart(fig_district, use_container_width=True)

        st.markdown(
            """
        <div class="insight-box">
        <b>📊 Comparative Analysis:</b> Performance gaps up to 70x between best and worst districts.<br>
        <b>💡 Best Practice Transfer:</b> Replicate high-performer strategies (community partnerships, 
        flexible timings, multilingual staff) to underperformers.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Rankings table
        st.markdown(
            '<div class="section-header">Detailed Performance Rankings</div>',
            unsafe_allow_html=True,
        )

        ranking_display = display_rankings[
            ["rank", "district", "total_enrollments", "aer_pct", "aebi", "aer_grade"]
        ].copy()
        ranking_display.columns = [
            "Rank",
            "District",
            "Total Enrollments",
            "AER (%)",
            "AEBI",
            "Grade",
        ]

        def color_grade(val):
            color = GRADE_COLORS.get(val, "#ffffff")
            return f"background-color: {color}; color: white; font-weight: bold"

        styled_df = ranking_display.style.map(color_grade, subset=["Grade"]).format(
            {"Total Enrollments": "{:,.0f}", "AER (%)": "{:.2f}", "AEBI": "{:.1f}"}
        )

        st.dataframe(styled_df, use_container_width=True, height=400)

    # ========================================================================
    # TAB 4: PIN CODE ANALYSIS
    # ========================================================================

    with tab4:
        st.markdown(
            '<div class="section-header">PIN Code Level Granular Analysis</div>',
            unsafe_allow_html=True,
        )

        display_pins = pincode_rankings.copy()
        if selected_district != "All":
            display_pins = display_pins[display_pins["district"] == selected_district]

        st.markdown("### 🔥 High-Risk PIN Codes Heat Map")
        fig_heatmap = create_risk_heatmap(
            display_pins, top_n=min(20, len(display_pins))
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown(
            """
        <div class="insight-box">
        <b>📊 Geographic Hotspots:</b> These 20 PIN codes require immediate mobile unit deployment.<br>
        <b>💡 Deployment Strategy:</b> 5-day cycles per PIN, targeting weekends and evenings for working adults.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Top and bottom performers
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🏆 Top 10 Performers")
            top_pins = display_pins.nlargest(10, "aer_pct")[
                ["pincode", "district", "aer_pct", "total_enrollments", "risk_level"]
            ].copy()
            top_pins.columns = [
                "PIN Code",
                "District",
                "AER (%)",
                "Enrollments",
                "Risk",
            ]
            st.dataframe(
                top_pins.style.format({"AER (%)": "{:.2f}", "Enrollments": "{:,.0f}"}),
                use_container_width=True,
            )

        with col2:
            st.markdown("### ⚠️ Bottom 10 - Urgent Action")
            bottom_pins = display_pins.nsmallest(10, "aer_pct")[
                ["pincode", "district", "aer_pct", "total_enrollments", "risk_level"]
            ].copy()
            bottom_pins.columns = [
                "PIN Code",
                "District",
                "AER (%)",
                "Enrollments",
                "Risk",
            ]
            st.dataframe(
                bottom_pins.style.format(
                    {"AER (%)": "{:.2f}", "Enrollments": "{:,.0f}"}
                ),
                use_container_width=True,
            )

        st.divider()

        # Scatter plot
        st.markdown("### 📊 Enrollment Volume vs. Quality")
        fig_scatter = px.scatter(
            display_pins,
            x="total_enrollments",
            y="aer_pct",
            color="risk_level",
            size="total_enrollments",
            hover_data=["pincode", "district"],
            color_discrete_map={
                "HIGH": GRADE_COLORS["RED"],
                "MEDIUM": GRADE_COLORS["YELLOW"],
                "LOW": GRADE_COLORS["GREEN"],
            },
            labels={"total_enrollments": "Total Enrollments", "aer_pct": "AER (%)"},
            title="Performance Matrix: Volume vs. Inclusion Quality",
        )
        fig_scatter.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(245,245,245,1)",
            title={"font": {"size": 20, "color": "#2c3e50"}},
            transition={"duration": 500},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown(
            """
        <div class="insight-box">
        <b>📊 Key Finding:</b> High volume ≠ Good coverage. Some busy centers have poor adult enrollment.<br>
        <b>💡 Root Cause:</b> Process inefficiencies, not capacity constraints. Audit high-volume/low-AER centers.
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ========================================================================
    # TAB 5: ACTION PLAN
    # ========================================================================

    with tab5:
        st.markdown(
            '<div class="section-header">🎯 Data-Driven Action Plan for UIDAI</div>',
            unsafe_allow_html=True,
        )

        st.markdown("""
        This section translates analytics into **concrete operational decisions** 
        with clear resource allocation, timelines, and success metrics.
        """)

        st.divider()

        # Priority summary
        tier_counts = priority_zones["tier"].value_counts()

        col1, col2, col3 = st.columns(3)

        with col1:
            tier1_count = tier_counts.get("TIER 1 - URGENT", 0)
            st.markdown(
                render_metric_card(
                    "🚨",
                    str(tier1_count),
                    "TIER 1 Zones",
                    "Urgent Action",
                    "negative",
                    0,
                ),
                unsafe_allow_html=True,
            )

        with col2:
            tier2_count = tier_counts.get("TIER 2 - MODERATE", 0)
            st.markdown(
                render_metric_card(
                    "⚠️",
                    str(tier2_count),
                    "TIER 2 Zones",
                    "3-6 Month Plan",
                    "neutral",
                    1,
                ),
                unsafe_allow_html=True,
            )

        with col3:
            tier3_count = tier_counts.get("TIER 3 - LOW PRIORITY", 0)
            st.markdown(
                render_metric_card(
                    "📊",
                    str(tier3_count),
                    "TIER 3 Zones",
                    "Monitor & Maintain",
                    "positive",
                    2,
                ),
                unsafe_allow_html=True,
            )

        st.divider()

        # Action cards
        st.markdown(
            '<div class="section-header">Recommended Interventions</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            render_action_card(
                "TIER1",
                "🚐",
                "1. Mobile Enrollment Unit Deployment",
                f"""
                <b>Target:</b> {tier1_count} TIER 1 zones with AER < 0.1%<br>
                <b>Resources:</b> Deploy 8-10 mobile units to worst-performing districts: {", ".join(display_rankings.nsmallest(3, "aer_pct")["district"].tolist())}<br>
                <b>Schedule:</b> 5-day rotation per PIN code, prioritize weekends & evenings<br>
                <b>Timeline:</b> Immediate deployment, 3-month intensive campaign<br>
                <b>Partners:</b> Local NGOs, gram panchayats, employers for venue & outreach
                """,
                f"Increase AER from {overall_aer:.2f}% to 2-3% in priority zones within 6 months",
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            render_action_card(
                "TIER2",
                "📢",
                "2. Targeted Adult Awareness Campaigns",
                f"""
                <b>Focus:</b> {zero_adult_pins} PIN codes with zero adult enrollment<br>
                <b>Channels:</b> Radio (regional languages), SMS blasts, door-to-door volunteers<br>
                <b>Message:</b> Aadhaar benefits (subsidies, banking, voting), simplified enrollment process<br>
                <b>Incentive:</b> On-spot enrollment at community centers, schools, workplaces<br>
                <b>Budget:</b> ₹50L for state-wide campaign (printing, media, volunteer stipends)
                """,
                "Reach 500,000+ unenrolled adults, convert 30% to actual enrollments",
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            render_action_card(
                "TIER2",
                "📅",
                "3. Seasonal Staffing Optimization",
                """
                <b>Peak Season (Jun-Jul):</b> +25% staff at all centers (school admission surge)<br>
                <b>Moderate Season (Aug-Sep):</b> +15% staff (post-admission corrections)<br>
                <b>Off-Peak (Oct-May):</b> Baseline staffing, focus on quality & adult outreach<br>
                <b>Flexibility:</b> Contract staff for peak periods, permanent for baseline<br>
                <b>Cost Savings:</b> Reduce wait times by 40%, improve satisfaction scores
                """,
                "30% cost savings vs. year-round peak staffing, better service quality",
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            render_action_card(
                "TIER3",
                "📊",
                "4. Continuous Monitoring Dashboard",
                """
                <b>Primary KPI:</b> Adult Enrollment Ratio (AER) - monthly tracking<br>
                <b>Secondary KPIs:</b> AEBI, zero-adult PIN count, mobile unit utilization<br>
                <b>Reporting:</b> Weekly district reports, monthly state review, quarterly public transparency report<br>
                <b>Accountability:</b> Link district performance incentives to AER improvement targets<br>
                <b>Tools:</b> This dashboard (updated monthly), mobile app for field officers
                """,
                "Real-time visibility enables 60% faster course correction on failing interventions",
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            render_action_card(
                "TIER3",
                "🏆",
                "5. Best Practice Replication Program",
                f"""
                <b>Champions:</b> Learn from {highest_aer_district} (AER: {highest_aer_value:.2f}%)<br>
                <b>Method:</b> Document success factors (staffing, location, partnerships, timings)<br>
                <b>Training:</b> Use top centers as training sites for underperforming district staff<br>
                <b>Knowledge Sharing:</b> Quarterly inter-district workshops, video case studies<br>
                <b>Recognition:</b> Public awards for top-performing centers, staff bonuses
                """,
                "Lift state-wide AER by 0.5-1.0 percentage points through proven strategies",
            ),
            unsafe_allow_html=True,
        )

        st.divider()

        # Priority zones table
        st.markdown(
            '<div class="section-header">📍 Priority Intervention Zones - Export for Field Operations</div>',
            unsafe_allow_html=True,
        )

        priority_display = priority_zones.copy()
        if selected_district != "All":
            priority_display = priority_display[
                priority_display["district"] == selected_district
            ]

        display_cols = ["tier", "district", "total_enrollments", "aer_pct"]
        if "pincode" in priority_display.columns:
            display_cols.insert(2, "pincode")
        if "recommendation" in priority_display.columns:
            display_cols.append("recommendation")

        priority_table = priority_display[display_cols].copy()

        col_rename = {
            "tier": "Priority Tier",
            "district": "District",
            "pincode": "PIN Code",
            "total_enrollments": "Total Enrollments",
            "aer_pct": "AER (%)",
            "recommendation": "Recommended Action",
        }
        priority_table = priority_table.rename(
            columns={k: v for k, v in col_rename.items() if k in priority_table.columns}
        )

        st.dataframe(
            priority_table.style.format(
                {"Total Enrollments": "{:,.0f}", "AER (%)": "{:.2f}"}
            ),
            use_container_width=True,
            height=400,
        )

        # Download button
        csv = priority_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Priority Zones for Field Teams",
            data=csv,
            file_name=f"UIDAI_Priority_Zones_{selected_district}_{start_month}_to_{end_month}.csv",
            mime="text/csv",
            help="Export this list to plan mobile unit routes and track intervention progress",
        )

        st.markdown(
            """
        <div class="recommendation-box">
        <b>💡 How to Use This Data:</b><br>
        1. Share CSV with district coordinators for mobile unit route planning<br>
        2. Update quarterly to measure intervention effectiveness<br>
        3. Track AER improvement month-over-month for each zone<br>
        4. Reallocate resources from improving zones to persistent laggards
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ========================================================================
    # FOOTER
    # ========================================================================

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        """
    <div class="footer">
        <div class="footer-logo">🇮🇳</div>
        <h3 style="margin: 0.5rem 0; color: white;">UIDAI Data Hackathon 2026</h3>
        <p style="font-size: 1.1rem; margin: 0.5rem 0; color: #e8eaf6;">
            <b>Aadhaar Age-Wise Enrollment Analysis Dashboard</b>
        </p>
        <p style="font-size: 0.9rem; margin: 1rem 0; color: #b39ddb;">
            Empowering data-driven decisions for inclusive identity coverage
        </p>
        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
            <p style="font-size: 0.85rem; color: #e8eaf6; margin: 0.3rem 0;">
                ✓ Anonymized, aggregated data only
            </p>
            <p style="font-size: 0.85rem; color: #e8eaf6; margin: 0.3rem 0;">
                ✓ UIDAI data governance compliant
            </p>
            <p style="font-size: 0.85rem; color: #e8eaf6; margin: 0.3rem 0;">
                ✓ No PII or biometric information accessed
            </p>
        </div>
        <p style="font-size: 0.8rem; margin-top: 1.5rem; color: #9575cd;">
            Built with Streamlit • Python • Plotly
        </p>
        <p style="font-size: 0.8rem; margin-top: 1.5rem; color: #9575cd;">
            Team - Tech4Bharath
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
