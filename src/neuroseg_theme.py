"""
NeuroSeg - Burgundy Glass Theme Module
Provides pure burgundy styling, removing all default Streamlit glare and washed-out highlights.
"""

from typing import Dict, Tuple, Any
# pyrefly: ignore[missing-import]
import streamlit as st

# -----------------------------------------------------------------------------
# Color Palette Tokens
# -----------------------------------------------------------------------------
COLORS = {
    "plum": "#4B2C3A",
    "plum_dark": "#2A1722",
    "navy_dark": "#180A12",
    "navy_subtle": "#1F0F18",
    "sky": "#E8B4C8",
    "pink": "#D9B3C4",
    "cream": "#F2E1D4",
    "royal_blue": "#9E3352",
    "mauve": "#381928",
}

# -----------------------------------------------------------------------------
# Mask Colors (RGB 0-255 format for slice overlay blending)
# -----------------------------------------------------------------------------
MASK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "WT": (232, 180, 200),
    "TC": (217, 179, 196),
    "ET": (242, 225, 212),
    "Whole Tumor": (232, 180, 200),
    "Tumor Core": (217, 179, 196),
    "Enhancing Tumor": (242, 225, 212),
    "Edema": (232, 180, 200),
    "Necrosis": (94, 35, 54),
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
        "linecolor": "rgba(242, 225, 212, 0.16)",
        "zerolinecolor": "rgba(242, 225, 212, 0.16)",
    },
    "yaxis": {
        "gridcolor": "rgba(242, 225, 212, 0.08)",
        "linecolor": "rgba(242, 225, 212, 0.16)",
        "zerolinecolor": "rgba(242, 225, 212, 0.16)",
    },
}

# -----------------------------------------------------------------------------
# Custom CSS Injection
# -----------------------------------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* 1. COMPLETELY KILL THE WHITE / GRAY PSEUDO-ELEMENT LIGHT GLARE */
.stApp::before,
.stApp::after,
.main::before,
.main::after,
[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after {
    display: none !important;
    background: none !important;
    content: none !important;
}

/* 2. SOLID BURGUNDY CANVAS ACROSS ALL WRAPPERS */
html, body, .stApp, .main,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stVerticalBlock"] {
    background-color: #24121C !important;
    background: #24121C !important;
    background-image: none !important;
    color: #F2E1D4 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

[data-testid="stHeader"] {
    background-color: rgba(36, 18, 28, 0.95) !important;
    backdrop-filter: blur(8px) !important;
}

/* 3. SIDEBAR STYLING & LABELS */
section[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: #1A0B14 !important;
    background: #1A0B14 !important;
    border-right: 1px solid rgba(242, 225, 212, 0.12) !important;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #F2E1D4 !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #F2E1D4 !important;
}

/* 4. FIX TAB LABELS & REMOVE HORIZONTAL CLIPPING */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px !important;
    background-color: #1A0B14 !important;
    padding: 6px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(242, 225, 212, 0.12) !important;
}

/* Force all tab titles to be bright cream so inactive tabs don't vanish */
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 16px !important;
    background: transparent !important;
    border: none !important;
}

.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] div,
.stTabs [data-baseweb="tab"] span {
    color: #F2E1D4 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    opacity: 0.75 !important;
}

.stTabs [data-baseweb="tab"]:hover p,
.stTabs [data-baseweb="tab"]:hover span {
    opacity: 1.0 !important;
    color: #FFFFFF !important;
}

.stTabs [aria-selected="true"] {
    background-color: #4B2C3A !important;
    border-bottom: 2px solid #D9B3C4 !important;
}

.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    opacity: 1.0 !important;
}

/* 5. SLIDER LABELS & CONTROLS */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] span {
    color: #F2E1D4 !important;
    font-weight: 600 !important;
    opacity: 0.95 !important;
}

.stSlider > div > div > div > div {
    background-color: #D9B3C4 !important;
}

/* 6. METRIC CARDS */
div[data-testid="stMetric"] {
    background: #3A1E2C !important;
    border: 1px solid rgba(242, 225, 212, 0.15) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
}

div[data-testid="stMetricLabel"] p {
    color: #F2E1D4 !important;
    opacity: 0.8 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* 7. DROPDOWNS & SELECTBOXES */
div[data-baseweb="select"] > div {
    background-color: #1A0B14 !important;
    border: 1px solid rgba(242, 225, 212, 0.2) !important;
    color: #F2E1D4 !important;
}

div[data-baseweb="popover"], div[data-baseweb="menu"] {
    background-color: #1A0B14 !important;
    border: 1px solid rgba(242, 225, 212, 0.2) !important;
}

div[data-baseweb="menu"] li {
    color: #F2E1D4 !important;
}

div[data-baseweb="menu"] li:hover {
    background-color: #4B2C3A !important;
}

/* 8. HERO CARD */
.ns-hero-card {
    background: #3A1E2C !important;
    border: 1px solid rgba(242, 225, 212, 0.18) !important;
    border-radius: 16px !important;
    padding: 22px 28px !important;
    margin-bottom: 20px !important;
}

.ns-title {
    font-size: 32px;
    font-weight: 800;
    color: #FFFFFF !important;
    margin: 0;
}

.ns-subtitle {
    font-size: 15px;
    font-weight: 600;
    color: #F2E1D4 !important;
    margin-top: 6px;
}

.ns-tagline {
    font-size: 12.5px;
    color: #F2E1D4 !important;
    opacity: 0.75;
    margin-top: 6px;
}

.ns-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    background: #4B2C3A;
    color: #F2E1D4;
    border: 1px solid rgba(217, 179, 196, 0.4);
    text-transform: uppercase;
}

.ns-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4ADE80;
    display: inline-block;
    margin-right: 5px;
}

.ns-pill-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 10px;
    font-weight: 700;
    background: rgba(217, 179, 196, 0.25);
    color: #D9B3C4;
    border: 1px solid rgba(217, 179, 196, 0.4);
    text-transform: uppercase;
    vertical-align: middle;
}
</style>
"""


def apply_theme() -> None:
    """Injects burgundy custom CSS styling into Streamlit."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def hero_header(title: str, subtitle: str, tag: str = "") -> None:
    """Renders a clinical hero header banner with burgundy styling."""
    tag_html = f'<div class="ns-tagline">{tag}</div>' if tag else ""
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