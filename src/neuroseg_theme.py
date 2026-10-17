"""
NeuroSeg - Royal Glass Theme Module
Provides high-fidelity royal glassmorphism styling, custom CSS tokens,
hero banners, Plotly chart configurations, and color palettes
(plum / navy / sky / pink / cream).
"""

from typing import Dict, Tuple, Any
# pyrefly: ignore[missing-import]
import streamlit as st

# -----------------------------------------------------------------------------
# Color Palette Tokens
# -----------------------------------------------------------------------------
# Royal Glass Theme:
# - Plum:   #4B2C3A
# - Navy:   #12162A / #1A1024
# - Sky:    #A7C6ED (RGB: 167, 198, 237)
# - Pink:   #D9B3C4 (RGB: 217, 179, 196)
# - Cream:  #F2E1D4 (RGB: 242, 225, 212)
# - Accents:#5B84CE, #8E5F74

COLORS = {
    "plum": "#4B2C3A",
    "plum_dark": "#2D1522",
    "navy_dark": "#0D0B14",
    "navy_subtle": "#161124",
    "sky": "#A7C6ED",
    "pink": "#D9B3C4",
    "cream": "#F2E1D4",
    "royal_blue": "#5B84CE",
    "mauve": "#8E5F74",
}

# -----------------------------------------------------------------------------
# Mask Colors (RGB 0-255 format for slice overlay blending)
# -----------------------------------------------------------------------------
MASK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "WT": (167, 198, 237),   # Sky blue for Whole Tumor (WT / Peritumoral Edema)
    "TC": (217, 179, 196),   # Pink for Tumor Core (TC)
    "ET": (242, 225, 212),   # Cream for Enhancing Active Tumor (ET)
    "Whole Tumor": (167, 198, 237),
    "Tumor Core": (217, 179, 196),
    "Enhancing Tumor": (242, 225, 212),
    "Edema": (167, 198, 237),
    "Necrosis": (142, 95, 116),
}

# -----------------------------------------------------------------------------
# Plotly Chart Styling
# -----------------------------------------------------------------------------
PLOT: Dict[str, Any] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {
        "color": "#F2E1D4",
        "family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "size": 12,
    },
    "xaxis": {
        "gridcolor": "rgba(242, 225, 212, 0.08)",
        "linecolor": "rgba(242, 225, 212, 0.15)",
        "zerolinecolor": "rgba(242, 225, 212, 0.15)",
    },
    "yaxis": {
        "gridcolor": "rgba(242, 225, 212, 0.08)",
        "linecolor": "rgba(242, 225, 212, 0.15)",
        "zerolinecolor": "rgba(242, 225, 212, 0.15)",
    },
}

# -----------------------------------------------------------------------------
# Custom CSS Injection
# -----------------------------------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Main App Background & Typography */
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 15%, rgba(75, 44, 58, 0.35) 0%, rgba(13, 11, 20, 1) 50%, rgba(18, 22, 42, 0.4) 100%) fixed !important;
    color: #F2E1D4 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

[data-testid="stHeader"] {
    background-color: rgba(13, 11, 20, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
}

/* Sidebar Glassmorphism */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(26, 16, 36, 0.92) 0%, rgba(18, 14, 28, 0.95) 100%) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-right: 1px solid rgba(242, 225, 212, 0.1) !important;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #F2E1D4 !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
}

/* Glass Cards for Metrics */
div[data-testid="stMetric"] {
    background: rgba(45, 28, 48, 0.45) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(242, 225, 212, 0.12) !important;
    border-radius: 12px !important;
    padding: 12px 18px !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25) !important;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(167, 198, 237, 0.35) !important;
    box-shadow: 0 8px 24px rgba(91, 132, 206, 0.25) !important;
}

div[data-testid="stMetricLabel"] {
    color: rgba(242, 225, 212, 0.75) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    text-transform: uppercase !important;
}

div[data-testid="stMetricValue"] {
    color: #F2E1D4 !important;
    font-weight: 700 !important;
    font-size: 22px !important;
}

div[data-testid="stMetricDelta"] {
    font-weight: 600 !important;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: rgba(26, 16, 36, 0.5) !important;
    padding: 6px 8px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(242, 225, 212, 0.08) !important;
    backdrop-filter: blur(10px) !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 16px !important;
    color: rgba(242, 225, 212, 0.7) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #FFFFFF !important;
    background-color: rgba(75, 44, 58, 0.35) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(75, 44, 58, 0.7) 0%, rgba(91, 132, 206, 0.3) 100%) !important;
    color: #F2E1D4 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #A7C6ED !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}

/* Buttons */
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"], button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #4B2C3A 0%, #5B84CE 100%) !important;
    color: #F2E1D4 !important;
    border: 1px solid rgba(242, 225, 212, 0.3) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(91, 132, 206, 0.3) !important;
    transition: all 0.2s ease !important;
}

.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(91, 132, 206, 0.5) !important;
    border-color: #A7C6ED !important;
}

.stButton > button[kind="secondary"], .stDownloadButton > button[kind="secondary"], button[data-testid="baseButton-secondary"] {
    background: rgba(45, 28, 48, 0.4) !important;
    color: #F2E1D4 !important;
    border: 1px solid rgba(242, 225, 212, 0.15) !important;
    border-radius: 8px !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.2s ease !important;
}

.stButton > button[kind="secondary"]:hover, .stDownloadButton > button[kind="secondary"]:hover, button[data-testid="baseButton-secondary"]:hover {
    background: rgba(75, 44, 58, 0.6) !important;
    border-color: rgba(242, 225, 212, 0.35) !important;
}

/* Form Controls & Inputs */
div[data-baseweb="select"] > div {
    background-color: rgba(30, 20, 38, 0.75) !important;
    border: 1px solid rgba(242, 225, 212, 0.15) !important;
    border-radius: 8px !important;
    color: #F2E1D4 !important;
}

div[data-baseweb="select"] span {
    color: #F2E1D4 !important;
}

.stSlider > div > div > div > div {
    background-color: #5B84CE !important;
}

/* Glass Dataframes & Tables */
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    background: rgba(35, 22, 42, 0.5) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(242, 225, 212, 0.12) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Custom Theme Badges & Components */
.ns-hero-card {
    background: linear-gradient(135deg, rgba(75, 44, 58, 0.55) 0%, rgba(26, 16, 36, 0.65) 55%, rgba(18, 24, 48, 0.55) 100%);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(242, 225, 212, 0.18);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 22px;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(242, 225, 212, 0.15);
}

.ns-title {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #FFFFFF 0%, #F2E1D4 40%, #D9B3C4 70%, #A7C6ED 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
    display: inline-block;
}

.ns-subtitle {
    font-size: 15px;
    font-weight: 600;
    color: #F2E1D4;
    margin-top: 5px;
    letter-spacing: 0.1px;
}

.ns-tagline {
    font-size: 12.5px;
    color: rgba(242, 225, 212, 0.72);
    margin-top: 7px;
    line-height: 1.4;
}

.ns-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    background: rgba(91, 132, 206, 0.18);
    color: #A7C6ED;
    border: 1px solid rgba(167, 198, 237, 0.35);
    backdrop-filter: blur(8px);
    box-shadow: 0 2px 8px rgba(91, 132, 206, 0.15);
    text-transform: uppercase;
}

.ns-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 8px #34d399;
    display: inline-block;
    margin-right: 5px;
    animation: ns-pulse 2s infinite ease-in-out;
}

.ns-pill-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
    background: rgba(217, 179, 196, 0.2);
    color: #D9B3C4;
    border: 1px solid rgba(217, 179, 196, 0.35);
    text-transform: uppercase;
    vertical-align: middle;
}

@keyframes ns-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}

/* Alert Boxes Styling */
div[data-testid="stAlert"] {
    background: rgba(45, 28, 48, 0.5) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(242, 225, 212, 0.15) !important;
    border-radius: 10px !important;
    color: #F2E1D4 !important;
}
</style>
"""


def apply_theme() -> None:
    """Injects royal glass custom CSS styling into Streamlit."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def hero_header(title: str, subtitle: str, tag: str = "") -> None:
    """Renders a clinical hero header banner with royal glass styling."""
    tag_html = f'<div class="ns-tagline">{tag}</div>' if tag else ''
    html_code = f"""
    <div class="ns-hero-card">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
            <h1 class="ns-title">{title}</h1>
            <span class="ns-pill-badge">CLINICAL WORKSTATION</span>
        </div>
        <div class="ns-subtitle">{subtitle}</div>
        {tag_html}
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)
