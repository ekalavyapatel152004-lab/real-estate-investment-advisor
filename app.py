from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# Real Estate Investment Advisor - Streamlit dashboard
# Place this file in the project ROOT beside data/, models/,
# outputs/, and src/.
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "outputs" / "cleaned_housing_data.csv",
    ROOT / "data" / "india_housing_prices.csv",
]
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional styling
st.markdown(
    """
    <style>
    /* ===== Global dashboard contrast fix (Streamlit/BaseWeb compatible) ===== */
    :root {
        color-scheme: light;
    }
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background-color: #f3f6fb !important;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
    }

    /* Force readable dark text across every page's main content. */
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] h5,
    [data-testid="stMain"] p,
    [data-testid="stMain"] li,
    [data-testid="stMain"] label,
    [data-testid="stMain"] span,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"],
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stMain"] [data-testid="stWidgetLabel"],
    [data-testid="stMain"] [data-testid="stWidgetLabel"] *,
    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] [data-testid="stCaptionContainer"] * {
        color: #17243b !important;
    }
    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] [data-testid="stCaptionContainer"] * {
        color: #52627a !important;
    }

    /* Inputs, select boxes, multiselects, number inputs and text areas. */
    [data-testid="stMain"] input,
    [data-testid="stMain"] textarea,
    [data-testid="stMain"] [data-baseweb="select"] *,
    [data-testid="stMain"] [data-baseweb="input"] *,
    [data-testid="stMain"] [data-baseweb="textarea"] *,
    [data-testid="stMain"] [data-baseweb="tag"] {
        color: #17243b !important;
        -webkit-text-fill-color: #17243b !important;
    }
    [data-testid="stMain"] [data-baseweb="select"] > div,
    [data-testid="stMain"] [data-baseweb="input"] > div,
    [data-testid="stMain"] [data-baseweb="textarea"] > div {
        background-color: #ffffff !important;
        border-color: #cbd5e1 !important;
    }
    /* Force every layer of main-area inputs to a light surface so the dark
       text is always readable, even when the browser/Streamlit theme is dark. */
    [data-testid="stMain"] [data-baseweb="select"],
    [data-testid="stMain"] [data-baseweb="select"] > div,
    [data-testid="stMain"] [data-baseweb="select"] > div > div,
    [data-testid="stMain"] [data-baseweb="input"],
    [data-testid="stMain"] [data-baseweb="base-input"],
    [data-testid="stMain"] [data-baseweb="input"] > div,
    [data-testid="stMain"] [data-testid="stNumberInputContainer"],
    [data-testid="stMain"] [data-testid="stNumberInput"] button,
    [data-testid="stMain"] [data-baseweb="select"] input,
    [data-testid="stMain"] [data-baseweb="base-input"] input {
        background-color: #ffffff !important;
        background: #ffffff !important;
    }
    [data-testid="stMain"] [data-baseweb="select"] > div,
    [data-testid="stMain"] [data-testid="stNumberInputContainer"] {
        border: 1px solid #cbd5e1 !important;
    }
    [data-testid="stMain"] [data-baseweb="select"] svg,
    [data-testid="stMain"] [data-testid="stNumberInput"] button svg {
        fill: #52627a !important;
        color: #52627a !important;
    }
    [data-testid="stMain"] input::placeholder {
        color: #7b8aa0 !important;
        -webkit-text-fill-color: #7b8aa0 !important;
    }

    /* Every layer inside main-area select boxes / number inputs: light surface,
       dark text (fixes the mix of white and dark-blue boxes). */
    [data-testid="stMain"] [data-testid="stSelectbox"] *,
    [data-testid="stMain"] [data-testid="stNumberInput"] *,
    [data-testid="stMain"] [data-testid="stTextInput"] *,
    [data-testid="stMain"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #17243b !important;
        -webkit-text-fill-color: #17243b !important;
    }
    [data-baseweb="popover"] ul, [data-baseweb="menu"] li {
        background-color: #ffffff !important;
    }
    [data-baseweb="menu"] li:hover, [role="option"]:hover {
        background-color: #e8eef8 !important;
    }

    /* Dropdown menus may be rendered outside the main container. */
    [role="listbox"], [role="option"],
    [data-baseweb="popover"], [data-baseweb="menu"] {
        color: #17243b !important;
        background-color: #ffffff !important;
    }
    [role="option"] * {
        color: #17243b !important;
    }

    /* Metrics and data cards. */
    [data-testid="stMain"] [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #dce5f0 !important;
        padding: 15px 16px !important;
        border-radius: 14px !important;
        box-shadow: 0 3px 12px rgba(18,38,63,.05) !important;
    }
    [data-testid="stMain"] [data-testid="stMetricLabel"],
    [data-testid="stMain"] [data-testid="stMetricLabel"] * {
        color: #52627a !important;
    }
    [data-testid="stMain"] [data-testid="stMetricValue"],
    [data-testid="stMain"] [data-testid="stMetricValue"] * {
        color: #14233d !important;
    }
    [data-testid="stMain"] [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    /* Sidebar keeps its dark background and white text. */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #101d35 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
        color: #f4f7fb !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #17243b !important;
        -webkit-text-fill-color: #17243b !important;
    }

    /* Buttons retain clear white text on their filled background. */
    [data-testid="stMain"] .stButton button,
    [data-testid="stMain"] .stDownloadButton button,
    [data-testid="stMain"] button[kind="primary"] {
        border-radius: 10px !important;
    }
    [data-testid="stMain"] .stButton button p,
    [data-testid="stMain"] .stDownloadButton button p,
    [data-testid="stMain"] button[kind="primary"] * {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- Professional real-estate theme (styling only) --------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    </style>
    <style>
    :root {
        --re-navy: #0b1b34;
        --re-navy-2: #14325f;
        --re-blue: #2563eb;
        --re-gold: #d4a537;
        --re-gold-2: #e9c75f;
        --re-ink: #17243b;
        --re-muted: #5b6b82;
        --re-line: #e1e8f2;
    }
    h1, h2, h3, h4, h5, p, li, label, button, input, textarea,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    html, body, .stApp,
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background:
            radial-gradient(circle at 100% 0%, rgba(37,99,235,.06), transparent 35%),
            radial-gradient(circle at 0% 100%, rgba(212,165,55,.07), transparent 35%),
            #f4f6fa !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stMainBlockContainer"] {
        padding-top: 3.2rem !important;
        max-width: 1500px;
    }

    /* ---------- Navbar ---------- */
    .st-key-topnav {
        background: linear-gradient(120deg, #0b1b34 0%, #12294d 100%) !important;
        border: 1px solid rgba(212,165,55,.35);
        border-radius: 16px;
        padding: 10px 14px;
        margin-bottom: 16px;
        box-shadow: 0 10px 28px rgba(11,27,52,.22);
        overflow: visible !important;
    }
    .st-key-topnav [data-testid="stHorizontalBlock"] { gap: 8px !important; align-items: center; }
    [data-testid="stMain"] .st-key-topnav button {
        width: 100% !important;
        min-height: 46px !important;
        height: auto !important;
        padding: 8px 10px !important;
        border-radius: 11px !important;
        background: rgba(255,255,255,.06) !important;
        border: 1px solid rgba(255,255,255,.14) !important;
        box-shadow: none !important;
        white-space: normal !important;
        overflow: visible !important;
        transition: all .18s ease;
    }
    [data-testid="stMain"] .st-key-topnav button:hover {
        background: rgba(255,255,255,.16) !important;
        transform: translateY(-1px);
    }
    [data-testid="stMain"] .st-key-topnav button *,
    [data-testid="stMain"] .st-key-topnav button p {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 600 !important;
        font-size: .92rem !important;
        line-height: 1.25 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    [data-testid="stMain"] .st-key-topnav button[kind="primary"],
    [data-testid="stMain"] .st-key-topnav button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #e9c75f, #c99a2e) !important;
        border-color: #e9c75f !important;
        box-shadow: 0 4px 14px rgba(212,165,55,.45) !important;
    }
    [data-testid="stMain"] .st-key-topnav button[kind="primary"] *,
    [data-testid="stMain"] .st-key-topnav button[kind="primary"] p,
    [data-testid="stMain"] .st-key-topnav button[data-testid="stBaseButton-primary"] *,
    [data-testid="stMain"] .st-key-topnav button[data-testid="stBaseButton-primary"] p {
        color: #0b1b34 !important;
        -webkit-text-fill-color: #0b1b34 !important;
        font-weight: 700 !important;
    }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-brand { display: flex; align-items: center; gap: 10px; }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-brand svg { flex: 0 0 auto; }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-brand-t1 {
        color: #ffffff !important; font-weight: 800; letter-spacing: .12em; font-size: .86rem; line-height: 1.1;
    }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-brand-t2 {
        color: #e9c75f !important; font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;
    }

    /* ---------- Hero ---------- */
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-hero {
        position: relative; overflow: hidden; border-radius: 22px;
        padding: 40px 44px; min-height: 230px; margin-bottom: 18px;
        background:
            radial-gradient(circle at 88% 15%, rgba(233,199,95,.30), transparent 42%),
            linear-gradient(120deg, #0b1b34 0%, #14325f 55%, #1f5aa8 100%);
        box-shadow: 0 18px 40px rgba(11,27,52,.28);
    }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-hero-text { position: relative; z-index: 2; max-width: 62%; }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-hero * { color: #ffffff !important; }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-hero .re-eyebrow {
        color: #e9c75f !important; font-weight: 700; font-size: .78rem; letter-spacing: .22em; margin-bottom: 10px;
    }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-hero h1 {
        font-family: 'Playfair Display', Georgia, serif; font-size: 2.6rem; line-height: 1.12;
        margin: 0 0 10px 0; padding: 0; font-weight: 700;
    }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-hero .re-sub {
        color: #d6e1f3 !important; font-size: 1.02rem; margin: 0 0 20px 0;
    }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-chips { display: flex; flex-wrap: wrap; gap: 10px; }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-chip {
        background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.25);
        padding: 7px 14px; border-radius: 999px; font-size: .86rem; font-weight: 600;
        backdrop-filter: blur(4px);
    }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-hero .re-chip b { color: #e9c75f !important; }
    .re-skyline { position: absolute; right: 0; bottom: 0; height: 100%; width: 55%; z-index: 1; opacity: .95; }

    /* ---------- Page banner ---------- */
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-banner {
        display: flex; align-items: center; gap: 16px;
        background: #ffffff; border: 1px solid var(--re-line); border-left: 5px solid #d4a537;
        border-radius: 16px; padding: 14px 20px; margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(18,38,63,.06);
    }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-banner-ic {
        width: 46px; height: 46px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem; background: linear-gradient(135deg, #0b1b34, #14325f);
    }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-banner-t { color: #0b1b34 !important; font-weight: 800; font-size: 1.12rem; }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-banner-d { color: #5b6b82 !important; font-size: .92rem; }

    /* ---------- Headings, cards, tables ---------- */
    [data-testid="stMain"] h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 1.7rem !important; font-weight: 700 !important;
        color: #0b1b34 !important; border-left: 5px solid #d4a537; padding-left: 14px !important;
        margin-top: .6rem !important;
    }
    [data-testid="stMain"] h4 { color: #0b1b34 !important; font-weight: 700 !important; font-size: 1.08rem !important; letter-spacing: .01em; }
    [data-testid="stMain"] [data-testid="stMetric"] {
        border-left: 5px solid #d4a537 !important;
        transition: transform .18s ease, box-shadow .18s ease;
    }
    [data-testid="stMain"] [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 26px rgba(18,38,63,.13) !important;
    }
    [data-testid="stMain"] [data-testid="stMetricValue"],
    [data-testid="stMain"] [data-testid="stMetricValue"] * { font-weight: 800 !important; }
    [data-testid="stMain"] [data-testid="stPlotlyChart"] {
        background: #ffffff; border: 1px solid var(--re-line); border-radius: 18px;
        padding: 12px 10px 4px 10px; box-shadow: 0 6px 20px rgba(18,38,63,.07);
        transition: box-shadow .18s ease;
    }
    [data-testid="stMain"] [data-testid="stPlotlyChart"]:hover { box-shadow: 0 14px 32px rgba(18,38,63,.13); }
    [data-testid="stMain"] [data-testid="stDataFrame"] {
        border: 1px solid var(--re-line); border-radius: 14px; overflow: hidden;
        box-shadow: 0 6px 20px rgba(18,38,63,.06);
    }
    [data-testid="stMain"] [data-testid="stAlert"] { border-radius: 14px !important; box-shadow: 0 4px 14px rgba(18,38,63,.06); }
    [data-testid="stMain"] hr { border-color: #dbe3ef !important; }
    [data-testid="stMain"] [data-baseweb="select"] > div,
    [data-testid="stMain"] [data-testid="stNumberInputContainer"] { border-radius: 10px !important; }
    [data-testid="stMain"] [data-testid="stForm"], [data-testid="stMain"] .st-key-advisor_form {
        background: #ffffff; border: 1px solid var(--re-line); border-radius: 18px;
        padding: 20px 22px; box-shadow: 0 6px 20px rgba(18,38,63,.06);
    }

    /* Main action buttons (not the navbar) */
    [data-testid="stMain"] button[kind="primary"],
    [data-testid="stMain"] button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #0b1b34, #2563eb) !important;
        border: none !important; padding: .55rem 1.6rem !important;
        box-shadow: 0 8px 20px rgba(37,99,235,.30) !important;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #0b1b34 0%, #12294d 100%) !important;
    }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(212,165,55,.35); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .re-side { display: flex; align-items: center; gap: 12px; padding: 6px 0 4px 0; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .re-side-t { color: #ffffff !important; font-weight: 800; letter-spacing: .1em; font-size: 1.02rem; line-height: 1.15; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .re-side-sub { color: #e9c75f !important; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; }
    html body [data-testid="stSidebar"] h3 {
        color: #e9c75f !important; font-size: .95rem !important; letter-spacing: .14em; text-transform: uppercase;
        border-bottom: 1px solid rgba(255,255,255,.15); padding-bottom: 8px;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] > div > div {
        background-color: #ffffff !important; border-radius: 10px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #17243b !important; -webkit-text-fill-color: #17243b !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"] { background: #e3ecfb !important; border-radius: 8px !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #52627a !important; }
    [data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
    [data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] { color: #ffffff !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #b9c6dc !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.15) !important; }

    /* ---------- Footer ---------- */
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-footer {
        margin-top: 28px; border-radius: 20px; padding: 26px 32px; text-align: center;
        background: linear-gradient(120deg, #0b1b34, #14325f);
        border-top: 3px solid #d4a537;
    }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-footer * { color: #cfdbef !important; }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-footer .re-foot-brand { color: #ffffff !important; font-weight: 800; letter-spacing: .08em; font-size: 1.05rem; margin-bottom: 6px; }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] .re-footer .re-foot-note { color: #e9c75f !important; font-size: .8rem; margin-top: 8px; }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: #b7c4da; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Helpers --------------------
def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None

def clean_text(series):
    return series.fillna("Unknown").astype(str)

def money_lakh(value):
    """Format prices stored in lakh rupees using Indian crore/lakh units."""
    if pd.isna(value):
        return "—"
    amount_lakh = float(value)
    if amount_lakh >= 100:
        return f"₹{amount_lakh / 100:,.2f} Cr"
    return f"₹{amount_lakh:,.2f} L"

def safe_numeric(frame, column):
    if column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame

def feature_row_for_model(model, row, dataset):
    """Align a one-row DataFrame to the feature names stored by the model."""
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None and hasattr(model, "named_steps"):
        # Pipeline usually exposes feature_names_in_ as well; this is a fallback.
        feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        names = list(feature_names)
        aligned = pd.DataFrame(index=[0])
        for col in names:
            if col in row.columns:
                aligned[col] = row[col].values
            elif col in dataset.columns:
                aligned[col] = dataset[col].iloc[0]
            else:
                aligned[col] = np.nan
        return aligned
    return row.copy()

def show_empty(message):
    st.info(message)

# -------------------- Load data --------------------
data_path = first_existing(DATA_CANDIDATES)
if data_path is None:
    st.error(
        "Housing dataset was not found. Please check that "
        "`outputs/cleaned_housing_data.csv` or "
        "`data/india_housing_prices.csv` exists inside your project folder."
    )
    st.stop()

df = pd.read_csv(data_path)
df.columns = [str(c).strip() for c in df.columns]

required_for_dashboard = ["City", "State", "Price_in_Lakhs", "Size_in_SqFt"]
missing_required = [c for c in required_for_dashboard if c not in df.columns]
if missing_required:
    st.error(f"Dataset is missing required columns: {', '.join(missing_required)}")
    st.stop()

for col in ["Price_in_Lakhs", "Price_per_SqFt", "Size_in_SqFt", "BHK",
            "Year_Built", "Age_of_Property", "Nearby_Schools",
            "Nearby_Hospitals", "Floor_No", "Total_Floors"]:
    safe_numeric(df, col)

# Always derive price per sq ft from the source price and area columns.
# Price_in_Lakhs is in lakh rupees, so multiply by 100,000 to get rupees.
# This also protects the dashboard when it falls back to the raw CSV, where
# Price_per_SqFt may be stored in lakh rupees per sq ft.
valid_area = df["Size_in_SqFt"].where(df["Size_in_SqFt"] > 0)
df["Price_per_SqFt"] = (
    df["Price_in_Lakhs"] * 100000 / valid_area
)
if "Age_of_Property" not in df.columns and "Year_Built" in df.columns:
    df["Age_of_Property"] = pd.Timestamp.today().year - df["Year_Built"]

for col in ["State", "City", "Locality", "Property_Type", "Furnished_Status",
            "Public_Transport_Accessibility", "Parking_Space", "Security",
            "Amenities", "Facing", "Owner_Type", "Availability_Status"]:
    if col in df.columns:
        df[col] = clean_text(df[col])

df = df.replace([np.inf, -np.inf], np.nan)
df = df[df["Price_in_Lakhs"].notna() & (df["Price_in_Lakhs"] > 0)].copy()
if df.empty:
    st.error("No rows with a valid positive property price were found.")
    st.stop()

# -------------------- Top navigation (horizontal) --------------------
PAGES = [
    "Executive Overview",
    "Property Explorer",
    "Market & EDA Insights",
    "Investment Comparison",
    "AI Investment Advisor",
    "Data Quality",
]
if st.session_state.get("nav_page") not in PAGES:
    st.session_state["nav_page"] = PAGES[0]

with st.container(key="topnav"):
    _nav_cols = st.columns([1.7] + [1.25] * len(PAGES))
    _nav_cols[0].markdown(
        '<div class="re-brand">'
        '<svg width="40" height="40" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="48" height="48" rx="12" fill="#d4a537"/>'
        '<path d="M10 24 L24 12 L38 24" fill="none" stroke="#0b1b34" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>'
        '<rect x="14" y="24" width="20" height="13" rx="1.5" fill="#0b1b34"/>'
        '<rect x="21" y="29" width="6" height="8" fill="#d4a537"/></svg>'
        '<div><div class="re-brand-t1">REAL ESTATE</div>'
        '<div class="re-brand-t2">Investment Advisor</div></div></div>',
        unsafe_allow_html=True,
    )
    for _col, _name in zip(_nav_cols[1:], PAGES):
        _is_active = st.session_state["nav_page"] == _name
        if _col.button(
            _name,
            key=f"nav_btn_{PAGES.index(_name)}",
            type="primary" if _is_active else "secondary",
        ):
            st.session_state["nav_page"] = _name
            st.rerun()
page = st.session_state["nav_page"]

# -------------------- Header --------------------
_SKYLINE = (
    '<svg class="re-skyline" viewBox="0 0 640 230" preserveAspectRatio="xMaxYMax meet" xmlns="http://www.w3.org/2000/svg">'
    '<defs><linearGradient id="reg" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0" stop-color="#ffffff" stop-opacity=".30"/><stop offset="1" stop-color="#ffffff" stop-opacity=".06"/></linearGradient>'
    '<pattern id="rew" width="14" height="16" patternUnits="userSpaceOnUse">'
    '<rect x="3" y="4" width="6" height="7" rx="1" fill="#f6d77a" fill-opacity=".6"/></pattern></defs>'
    '<g fill="url(#reg)">'
    '<rect x="20" y="120" width="70" height="110"/><rect x="100" y="60" width="60" height="170"/>'
    '<rect x="170" y="100" width="80" height="130"/><rect x="260" y="30" width="56" height="200"/>'
    '<rect x="326" y="80" width="90" height="150"/><rect x="426" y="130" width="64" height="100"/>'
    '<rect x="500" y="90" width="72" height="140"/></g>'
    '<g fill="url(#rew)">'
    '<rect x="20" y="120" width="70" height="110"/><rect x="100" y="60" width="60" height="170"/>'
    '<rect x="170" y="100" width="80" height="130"/><rect x="260" y="30" width="56" height="200"/>'
    '<rect x="326" y="80" width="90" height="150"/><rect x="426" y="130" width="64" height="100"/>'
    '<rect x="500" y="90" width="72" height="140"/></g>'
    '<line x1="288" y1="30" x2="288" y2="10" stroke="#e9c75f" stroke-width="2"/>'
    '<polygon points="580,230 580,188 608,160 636,188 636,230" fill="#d4a537"/>'
    '<rect x="601" y="200" width="14" height="30" fill="#0b1b34" fill-opacity=".7"/>'
    '<rect x="0" y="228" width="640" height="2" fill="#e9c75f"/></svg>'
)
st.markdown(
    '<div class="re-hero">' + _SKYLINE +
    '<div class="re-hero-text">'
    '<div class="re-eyebrow">PROPERTY MARKET INTELLIGENCE</div>'
    '<h1>🏡 Real Estate Investment Advisor</h1>'
    '<p class="re-sub">Property market intelligence • Interactive exploration • Investment screening</p>'
    '<div class="re-chips">'
    f'<span class="re-chip">🏠 <b>{len(df):,}</b> listings</span>'
    f'<span class="re-chip">🏙️ <b>{df["City"].nunique():,}</b> cities</span>'
    f'<span class="re-chip">🗺️ <b>{df["State"].nunique():,}</b> states</span>'
    '</div></div></div>',
    unsafe_allow_html=True,
)

_PAGE_INFO = {
    "Executive Overview": ("📊", "Headline market metrics and the big picture across your selected listings."),
    "Property Explorer": ("🏘️", "Browse, sort and inspect individual property listings."),
    "Market & EDA Insights": ("📈", "Price, size, age and location patterns across the market."),
    "Investment Comparison": ("⚖️", "Compare properties and spot value against city benchmarks."),
    "AI Investment Advisor": ("🤖", "Model-assisted screening and scenario-based price projection."),
    "Data Quality": ("🛡️", "Completeness and validation checks for the underlying dataset."),
}
_icon, _desc = _PAGE_INFO[page]
st.markdown(
    '<div class="re-banner">'
    f'<div class="re-banner-ic">{_icon}</div>'
    f'<div><div class="re-banner-t">{page}</div><div class="re-banner-d">{_desc}</div></div></div>',
    unsafe_allow_html=True,
)
st.warning(
    "Educational decision-support prototype only — not financial advice. "
    "The investment label is based on a relative city price-per-sq-ft rule, "
    "and future values are scenarios based on an assumed appreciation rate."
)

# -------------------- Sidebar navigation and filters --------------------
_LOGO_SVG = (
    '<svg width="38" height="38" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="48" height="48" rx="12" fill="#d4a537"/>'
    '<path d="M10 24 L24 12 L38 24" fill="none" stroke="#0b1b34" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>'
    '<rect x="14" y="24" width="20" height="13" rx="1.5" fill="#0b1b34"/>'
    '<rect x="21" y="29" width="6" height="8" fill="#d4a537"/></svg>'
)
st.sidebar.markdown(
    '<div class="re-side">' + _LOGO_SVG +
    '<div><div class="re-side-t">REAL ESTATE ANALYTICS</div>'
    '<div class="re-side-sub">Investment intelligence</div></div></div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Dashboard filters")

def options_for(col):
    """Return sorted options from the complete dashboard dataset."""
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().astype(str).unique().tolist())

def filtered_options(frame, col):
    """Return options available after the preceding filters."""
    if col not in frame.columns:
        return []
    return sorted(frame[col].dropna().astype(str).unique().tolist())

# Cascading geographic filters: State -> City -> Locality.
filtered = df.copy()

if "State" in filtered.columns:
    state_options = filtered_options(filtered, "State")
    selected_states = st.sidebar.multiselect("State", state_options, default=[])
    if selected_states:
        filtered = filtered[filtered["State"].astype(str).isin(selected_states)]

if "City" in filtered.columns:
    city_options = filtered_options(filtered, "City")
    selected_cities = st.sidebar.multiselect("City", city_options, default=[])
    if selected_cities:
        filtered = filtered[filtered["City"].astype(str).isin(selected_cities)]

if "Locality" in filtered.columns:
    locality_options = filtered_options(filtered, "Locality")
    selected_localities = st.sidebar.multiselect(
        "Locality", locality_options, default=[]
    )
    if selected_localities:
        filtered = filtered[
            filtered["Locality"].astype(str).isin(selected_localities)
        ]

# Other categorical filters are also narrowed by the geographic selections.
for col, label in [
    ("Property_Type", "Property type"),
    ("Furnished_Status", "Furnishing"),
    ("Availability_Status", "Availability"),
]:
    if col in filtered.columns:
        opts = filtered_options(filtered, col)
        chosen = st.sidebar.multiselect(label, opts, default=[], key=f"filter_{col}")
        if chosen:
            filtered = filtered[filtered[col].astype(str).isin(chosen)]

if "BHK" in filtered.columns:
    bhk_values = sorted(filtered["BHK"].dropna().unique().tolist())
    if bhk_values:
        chosen_bhk = st.sidebar.multiselect("BHK", bhk_values, default=[])
        if chosen_bhk:
            filtered = filtered[filtered["BHK"].isin(chosen_bhk)]

# Numeric sliders are only rendered when at least one row remains.
if not filtered.empty:
    price_min = float(filtered["Price_in_Lakhs"].min())
    price_max = float(filtered["Price_in_Lakhs"].max())
    if np.isfinite(price_min) and np.isfinite(price_max) and price_min < price_max:
        p_low, p_high = st.sidebar.slider(
            "Price range (₹ lakh)",
            min_value=price_min,
            max_value=price_max,
            value=(price_min, price_max),
        )
        filtered = filtered[filtered["Price_in_Lakhs"].between(p_low, p_high)]

if (
    not filtered.empty
    and "Size_in_SqFt" in filtered.columns
    and filtered["Size_in_SqFt"].notna().any()
):
    area_min = float(filtered["Size_in_SqFt"].min())
    area_max = float(filtered["Size_in_SqFt"].max())
    if np.isfinite(area_min) and np.isfinite(area_max) and area_min < area_max:
        a_low, a_high = st.sidebar.slider(
            "Area range (sq ft)",
            min_value=area_min,
            max_value=area_max,
            value=(area_min, area_max),
        )
        filtered = filtered[filtered["Size_in_SqFt"].between(a_low, a_high)]

st.sidebar.caption(f"{len(filtered):,} properties match the current filters.")

# These pages depend on the selected property subset. Never silently fall back
# to the complete dataset, because that would make the displayed results misleading.
if filtered.empty and page in {
    "Executive Overview",
    "Property Explorer",
    "Market & EDA Insights",
    "Investment Comparison",
}:
    st.warning(
        "No properties match the selected filters. Please clear or adjust one "
        "or more filters in the sidebar."
    )
    st.stop()

# -------------------- Page 1: Overview --------------------
if page == "Executive Overview":
    st.subheader("Market snapshot")
    view = filtered
    avg_price = view["Price_in_Lakhs"].mean()
    median_price = view["Price_in_Lakhs"].median()
    avg_ppsf = view["Price_per_SqFt"].mean() if "Price_per_SqFt" in view else np.nan
    if "Availability_Status" in view:
        # Treat common positive listing statuses as available, while excluding
        # phrases such as "not available" and "unavailable".
        status = view["Availability_Status"].fillna("").astype(str).str.strip().str.lower()
        positive_status = status.str.contains(
            r"available|ready[ _-]?to[ _-]?move|immediate|vacant",
            regex=True,
            na=False,
        )
        negative_status = status.str.contains(
            r"not[ _-]?available|unavailable|sold|booked",
            regex=True,
            na=False,
        )
        available_count = int((positive_status & ~negative_status).sum())
    else:
        available_count = np.nan
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total properties", f"{len(view):,}")
    k2.metric("Average price", money_lakh(avg_price))
    k3.metric("Median price", money_lakh(median_price))
    k4.metric("Average price / sq ft", f"₹{avg_ppsf:,.0f}" if pd.notna(avg_ppsf) else "—")
    k5.metric("Available properties", f"{available_count:,.0f}" if pd.notna(available_count) else "—")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Property price distribution")
        fig = px.histogram(view, x="Price_in_Lakhs", nbins=35,
                           labels={"Price_in_Lakhs": "Price (₹ lakh)"},
                           color_discrete_sequence=["#168c91"])
        fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### Average price by city")
        city_avg = view.groupby("City", as_index=False)["Price_in_Lakhs"].mean().nlargest(15, "Price_in_Lakhs")
        fig = px.bar(city_avg.sort_values("Price_in_Lakhs"), x="Price_in_Lakhs", y="City",
                     orientation="h", labels={"Price_in_Lakhs": "Average price (₹ lakh)"},
                     color="Price_in_Lakhs", color_continuous_scale="Teal")
        fig.update_layout(template="plotly_white", height=340, coloraxis_showscale=False,
                          margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Property count by state")
        state_counts = view.groupby("State").size().reset_index(name="Properties").nlargest(15, "Properties")
        fig = px.bar(state_counts.sort_values("Properties"), x="Properties", y="State",
                     orientation="h", color_discrete_sequence=["#3867a8"])
        fig.update_layout(template="plotly_white", height=330, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### Property type distribution")
        if "Property_Type" in view:
            types = view["Property_Type"].value_counts().rename_axis("Property type").reset_index(name="Count")
            fig = px.pie(types, names="Property type", values="Count", hole=.48,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(template="plotly_white", height=330, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")

# -------------------- Page 2: Explorer --------------------
elif page == "Property Explorer":
    st.subheader("Explore listed properties")
    if filtered.empty:
        show_empty("No properties match the selected filters. Clear one or more filters in the sidebar.")
    else:
        sort_col = st.selectbox(
            "Sort properties by",
            [c for c in ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"] if c in filtered.columns],
            format_func=lambda x: {
                "Price_in_Lakhs": "Price (₹ lakh)",
                "Size_in_SqFt": "Area (sq ft)",
                "Price_per_SqFt": "Price per sq ft (₹)",
            }.get(x, x),
        )
        sort_order = st.radio("Order", ["Highest first", "Lowest first"], horizontal=True)
        shown = filtered.sort_values(sort_col, ascending=(sort_order == "Lowest first"))
        display_cols = [c for c in [
            "ID", "State", "City", "Locality", "Property_Type", "BHK",
            "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
            "Furnished_Status", "Availability_Status"
        ] if c in shown.columns]
        st.dataframe(
            shown[display_cols].head(2000),
            width="stretch",
            hide_index=True,
            column_config={
                "Price_per_SqFt": st.column_config.NumberColumn(
                    "Price per SqFt (₹)", format="₹%,.2f"
                ),
                "Price_in_Lakhs": st.column_config.NumberColumn(
                    "Price (₹ lakh)", format="%.2f"
                ),
                "Size_in_SqFt": st.column_config.NumberColumn(
                    "Area (sq ft)", format="%,.0f"
                ),
            },
        )
        st.caption("Showing up to 2,000 matching rows in the table.")
        st.download_button(
            "Download filtered properties (CSV)",
            data=shown.to_csv(index=False).encode("utf-8"),
            file_name="filtered_housing_properties.csv",
            mime="text/csv",
        )

        st.markdown("#### View complete property details")
        if "ID" in shown.columns:
            selected_id = st.selectbox("Select property ID", shown["ID"].astype(str).tolist())
            selected = shown[shown["ID"].astype(str) == selected_id].iloc[0]
        else:
            selected_index = st.selectbox("Select row", shown.index.tolist())
            selected = shown.loc[selected_index]
        details = selected.dropna().astype(str).to_frame("Value")
        st.dataframe(details, width="stretch")

# -------------------- Page 3: EDA --------------------
elif page == "Market & EDA Insights":
    st.subheader("Market and exploratory data analysis")
    view = filtered

    if "Size_in_SqFt" in view.columns:
        st.markdown("#### Property price vs size")
        scatter = view.dropna(subset=["Size_in_SqFt", "Price_in_Lakhs"])
        fig = px.scatter(scatter.sample(min(len(scatter), 12000), random_state=42),
                         x="Size_in_SqFt", y="Price_in_Lakhs",
                         color="Property_Type" if "Property_Type" in scatter else None,
                         hover_data=[c for c in ["City", "Locality", "BHK"] if c in scatter],
                         labels={"Size_in_SqFt": "Size (sq ft)", "Price_in_Lakhs": "Price (₹ lakh)"},
                         opacity=.65)
        fig.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Price per sq ft by property type")
        if "Property_Type" in view and "Price_per_SqFt" in view:
            d = view.groupby("Property_Type", as_index=False)["Price_per_SqFt"].median()
            fig = px.bar(d, x="Property_Type", y="Price_per_SqFt", color="Property_Type",
                         labels={"Price_per_SqFt": "Median price per sq ft (₹)"})
            fig.update_layout(template="plotly_white", showlegend=False, height=350)
            st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("#### Top 5 expensive localities")
        if {"City", "Locality"}.issubset(view.columns):
            d = (
                view.groupby(["City", "Locality"], as_index=False)["Price_in_Lakhs"]
                .mean()
                .nlargest(5, "Price_in_Lakhs")
            )
            d["City / Locality"] = d["City"].astype(str) + " — " + d["Locality"].astype(str)
            fig = px.bar(
                d.sort_values("Price_in_Lakhs"),
                x="Price_in_Lakhs",
                y="City / Locality",
                orientation="h",
                labels={"Price_in_Lakhs": "Average price (₹ lakh)"},
            )
            fig.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### BHK distribution")
        if "BHK" in view:
            d = view["BHK"].value_counts().sort_index().rename_axis("BHK").reset_index(name="Properties")
            fig = px.bar(d, x="BHK", y="Properties", color_discrete_sequence=["#168c91"])
            fig.update_layout(template="plotly_white", height=340)
            st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("#### Property age analysis")
        if "Age_of_Property" in view:
            d = view.dropna(subset=["Age_of_Property", "Price_in_Lakhs"])
            fig = px.scatter(d.sample(min(len(d), 10000), random_state=42),
                             x="Age_of_Property", y="Price_in_Lakhs",
                             labels={"Age_of_Property": "Property age (years)", "Price_in_Lakhs": "Price (₹ lakh)"},
                             opacity=.55)
            fig.update_layout(template="plotly_white", height=340)
            st.plotly_chart(fig, width="stretch")

    st.markdown("#### Price per sq ft by state")
    if "State" in view and "Price_per_SqFt" in view:
        d = view.groupby("State", as_index=False)["Price_per_SqFt"].median().nlargest(15, "Price_per_SqFt")
        fig = px.bar(d.sort_values("Price_per_SqFt"), x="Price_per_SqFt", y="State", orientation="h",
                     labels={"Price_per_SqFt": "Median price per sq ft (₹)"})
        fig.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig, width="stretch")

    # Additional analyses requested in the project specification.
    st.markdown("#### Price and size distribution with outlier checks")
    e1, e2 = st.columns(2)
    with e1:
        fig = px.histogram(view, x="Size_in_SqFt", nbins=40, title="Property size distribution", color_discrete_sequence=["#287c8e"])
        fig.update_layout(template="plotly_white", height=330)
        st.plotly_chart(fig, width="stretch")
    with e2:
        fig = px.box(view, y="Price_per_SqFt", points="outliers", title="Price per sq ft (₹) — outlier view", color_discrete_sequence=["#e39b45"])
        fig.update_layout(template="plotly_white", height=330)
        st.plotly_chart(fig, width="stretch")

    e1, e2 = st.columns(2)
    with e1:
        if "State" in view and "Price_per_SqFt" in view:
            state_avg = view.groupby("State", as_index=False)["Price_per_SqFt"].mean().nlargest(15, "Price_per_SqFt")
            fig = px.bar(
                state_avg.sort_values("Price_per_SqFt"),
                x="Price_per_SqFt",
                y="State",
                orientation="h",
                title="Average price per sq ft (₹) by state",
                labels={"Price_per_SqFt": "Average price per sq ft (₹)"},
                color="Price_per_SqFt",
                color_continuous_scale="Teal",
            )
            fig.update_layout(template="plotly_white", height=390, coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
        if "Locality" in view and "Age_of_Property" in view:
            locality_age = view.groupby(["City", "Locality"], as_index=False)["Age_of_Property"].median().rename(columns={"Age_of_Property":"Median age (years)"}).nsmallest(20, "Median age (years)")
            st.markdown("**Median property age by locality (20 youngest)**")
            st.dataframe(locality_age, width="stretch", hide_index=True)
    with e2:
        if "City" in view and "BHK" in view:
            bhk_city = view.groupby(["City", "BHK"]).size().reset_index(name="Properties")
            top_cities = view["City"].value_counts().head(10).index
            bhk_city = bhk_city[bhk_city["City"].isin(top_cities)]
            fig = px.bar(bhk_city, x="City", y="Properties", color="BHK", title="BHK distribution across top 10 cities", barmode="stack")
            fig.update_layout(template="plotly_white", height=390, xaxis_tickangle=-35)
            st.plotly_chart(fig, width="stretch")
        if "Locality" in view:
            locality_price = view.groupby(["City", "Locality"], as_index=False)["Price_in_Lakhs"].mean().nlargest(5, "Price_in_Lakhs")
            st.markdown("**Five most expensive localities by average listing price**")
            st.dataframe(locality_price, width="stretch", hide_index=True)
            st.caption("The dataset has no listing-date history, so this is a locality price ranking, not a time-series price trend.")

    e1, e2 = st.columns(2)
    with e1:
        if "Owner_Type" in view:
            owner_counts = view["Owner_Type"].value_counts().rename_axis("Owner type").reset_index(name="Properties")
            fig = px.bar(owner_counts, x="Owner type", y="Properties", title="Properties by owner type", color_discrete_sequence=["#287c8e"])
            fig.update_layout(template="plotly_white", height=330)
            st.plotly_chart(fig, width="stretch")
        if "Parking_Space" in view:
            parking_price = view.groupby("Parking_Space", as_index=False)["Price_in_Lakhs"].median()
            fig = px.bar(parking_price, x="Parking_Space", y="Price_in_Lakhs", title="Median price by parking availability", color_discrete_sequence=["#6686bd"])
            fig.update_layout(template="plotly_white", height=330)
            st.plotly_chart(fig, width="stretch")
    with e2:
        if "Availability_Status" in view:
            avail_counts = view["Availability_Status"].value_counts().rename_axis("Availability status").reset_index(name="Properties")
            fig = px.pie(avail_counts, names="Availability status", values="Properties", hole=.45, title="Listings by availability status")
            fig.update_layout(template="plotly_white", height=330)
            st.plotly_chart(fig, width="stretch")
        if "Amenities" in view:
            amenity_price = view.groupby("Amenities", as_index=False)["Price_per_SqFt"].median().nlargest(12, "Price_per_SqFt")
            fig = px.bar(amenity_price, x="Price_per_SqFt", y="Amenities", orientation="h", title="Median price per sq ft (₹) by amenity group")
            fig.update_layout(template="plotly_white", height=330)
            st.plotly_chart(fig, width="stretch")

    st.markdown("#### Accessibility, furnishing and property features")
    categorical_candidates = [
        ("Nearby_Schools", "Nearby schools"),
        ("Nearby_Hospitals", "Nearby hospitals"),
        ("Furnished_Status", "Furnishing status"),
        ("Facing", "Facing direction"),
        ("Owner_Type", "Owner type"),
        ("Availability_Status", "Availability"),
        ("Parking_Space", "Parking"),
        ("Security", "Security"),
        ("Public_Transport_Accessibility", "Public transport"),
    ]
    available_candidates = [(c, label) for c, label in categorical_candidates if c in view.columns]
    if available_candidates:
        selected_feature = st.selectbox("Choose a feature to compare with price", available_candidates,
                                        format_func=lambda x: x[1])
        feature_col = selected_feature[0]
        temp = view.copy()
        if pd.api.types.is_numeric_dtype(temp[feature_col]):
            temp["Feature group"] = pd.cut(temp[feature_col], bins=5, duplicates="drop").astype(str)
        else:
            temp["Feature group"] = temp[feature_col].astype(str)
        summary = temp.groupby("Feature group", as_index=False, observed=False)["Price_in_Lakhs"].median()
        summary = summary.sort_values("Price_in_Lakhs", ascending=False).head(20)
        fig = px.bar(summary, x="Feature group", y="Price_in_Lakhs",
                     labels={"Feature group": selected_feature[1], "Price_in_Lakhs": "Median price (₹ lakh)"})
        fig.update_layout(template="plotly_white", height=380, xaxis_tickangle=-30)
        st.plotly_chart(fig, width="stretch")

    numeric_cols = view.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if view[c].nunique(dropna=True) > 1]
    if len(numeric_cols) >= 2:
        st.markdown("#### Numeric feature correlation")
        corr = view[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1)
        fig.update_layout(template="plotly_white", height=520)
        st.plotly_chart(fig, width="stretch")

# -------------------- Page 4: Investment comparison --------------------
elif page == "Investment Comparison":
    st.subheader("Compare properties and investment indicators")
    view = filtered
    if "Price_per_SqFt" in view.columns:
        st.markdown("#### Properties priced below their city median per sq ft")
        # Benchmark against each city's median from the complete valid dataset,
        # not the currently filtered subset.
        city_median_map = (
            df.dropna(subset=["City", "Price_per_SqFt"])
            .groupby("City")["Price_per_SqFt"]
            .median()
        )
        candidates = view.copy()
        candidates["City_Median_Price_Per_SqFt"] = candidates["City"].map(city_median_map)
        candidates["Below_City_Median"] = candidates["Price_per_SqFt"] < candidates["City_Median_Price_Per_SqFt"]
        below = candidates[candidates["Below_City_Median"]].sort_values("Price_per_SqFt").head(100)
        st.metric("Listings below their city's median price / sq ft", f"{int(candidates['Below_City_Median'].sum()):,}")
        show_cols = [c for c in ["ID", "City", "Locality", "Property_Type", "BHK",
                                 "Price_in_Lakhs", "Price_per_SqFt", "City_Median_Price_Per_SqFt"]
                     if c in below.columns]
        st.dataframe(
            below[show_cols],
            width="stretch",
            hide_index=True,
            column_config={
                "Price_per_SqFt": st.column_config.NumberColumn(
                    "Price per SqFt (₹)", format="₹%,.2f"
                ),
                "City_Median_Price_Per_SqFt": st.column_config.NumberColumn(
                    "City median (₹/SqFt)", format="₹%,.2f"
                ),
                "Price_in_Lakhs": st.column_config.NumberColumn(
                    "Price (₹ lakh)", format="%.2f"
                ),
            },
        )
        st.download_button("Download below-city-median listings",
                           below.to_csv(index=False).encode("utf-8"),
                           "below_city_median_properties.csv", "text/csv")

    st.markdown("#### Compare up to five properties")
    if "ID" in view.columns:
        st.caption(
            "For performance, enter IDs directly instead of loading all 250,000 IDs "
            "into a browser dropdown. You can find IDs in the Property Explorer."
        )
        entered_ids = st.text_input(
            "Enter up to five property IDs (comma-separated)",
            placeholder="Example: 217209, 245722",
            key="comparison_property_ids",
        )
        selected_ids = [item.strip() for item in entered_ids.split(",") if item.strip()][:5]
        if len([item for item in entered_ids.split(",") if item.strip()]) > 5:
            st.warning("Please enter no more than five property IDs.")
        if selected_ids:
            id_text = view["ID"].astype(str)
            comp = view[id_text.isin(selected_ids)].copy()
            found_ids = set(comp["ID"].astype(str))
            missing_ids = [item for item in selected_ids if item not in found_ids]
            if missing_ids:
                st.warning("These IDs were not found in the current filtered dataset: " + ", ".join(missing_ids))
            compare_cols = [c for c in [
                "ID", "State", "City", "Locality", "Property_Type", "BHK",
                "Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt",
                "Furnished_Status", "Amenities", "Parking_Space", "Security",
                "Public_Transport_Accessibility", "Availability_Status"
            ] if c in comp.columns]
            st.dataframe(
                comp[compare_cols].set_index("ID").T,
                width="stretch",
            )
            # Transparent heuristic only; not a trained return/profit score.
            score_rows = []
            for _, r in comp.iterrows():
                score = 0
                reasons = []
                if "Price_per_SqFt" in comp and pd.notna(r.get("Price_per_SqFt")):
                    city_med = city_median_map.get(r["City"], np.nan)
                    if pd.notna(city_med) and r["Price_per_SqFt"] < city_med:
                        score += 1
                        reasons.append("Below city median price/sq ft")
                if str(r.get("Parking_Space", "")).lower() in ["yes", "available", "true", "1"]:
                    score += 1
                    reasons.append("Parking indicated")
                if str(r.get("Security", "")).lower() not in ["no", "none", "unknown", "nan", ""]:
                    score += 1
                    reasons.append("Security information present")
                if str(r.get("Availability_Status", "")).lower() in ["available", "ready to move", "ready-to-move"]:
                    score += 1
                    reasons.append("Listed as available")
                score_rows.append({"Property ID": str(r.get("ID")), "Feature indicator (0–4)": score,
                                   "Reasons": "; ".join(reasons) if reasons else "No listed positive indicators"})
            st.markdown("#### Transparent feature indicators")
            st.dataframe(pd.DataFrame(score_rows), width="stretch", hide_index=True)
            st.caption("This simple feature indicator is not a probability of profit or a prediction of investment return.")
    else:
        show_empty("The dataset needs an ID column to select and compare individual properties.")

    if "Locality" in view.columns:
        st.markdown("#### Locality market ranking")
        agg_dict = {"Average_Price_Lakhs": ("Price_in_Lakhs", "mean"),
                    "Property_Count": ("Price_in_Lakhs", "count")}
        if "Price_per_SqFt" in view:
            agg_dict["Median_Price_Per_SqFt"] = ("Price_per_SqFt", "median")
        locality_rank = view.groupby(["City", "Locality"], as_index=False).agg(**agg_dict)
        st.dataframe(
            locality_rank.sort_values("Property_Count", ascending=False).head(30),
            width="stretch",
            hide_index=True,
            column_config={
                "Average_Price_Lakhs": st.column_config.NumberColumn(
                    "Average price (₹ lakh)", format="%.2f"
                ),
                "Median_Price_Per_SqFt": st.column_config.NumberColumn(
                    "Median price per SqFt (₹)", format="₹%,.2f"
                ),
            },
        )

# -------------------- Page 5: AI advisor --------------------
elif page == "AI Investment Advisor":
    st.subheader("AI-assisted property screening")
    st.write("Enter the property details below. The saved models are used if their expected feature columns can be matched.")
    cls_path = MODEL_DIR / "investment_classifier.joblib"
    reg_path = MODEL_DIR / "current_price_regressor.joblib"
    st.markdown("#### Model evaluation and feature importance")
    metrics_files = [("Classification metrics", ROOT / "outputs" / "classification_metrics.json"), ("Regression metrics", ROOT / "outputs" / "regression_metrics.json")]
    metric_cols = st.columns(2)
    for metric_col, (metric_title, metric_file) in zip(metric_cols, metrics_files):
        with metric_col:
            st.markdown(f"**{metric_title}**")
            if metric_file.exists():
                try:
                    st.json(json.loads(metric_file.read_text(encoding="utf-8")))
                except Exception:
                    st.caption("Metrics file could not be read.")
            else:
                st.caption("Run the training script to generate evaluation metrics.")
    cm_path = ROOT / "outputs" / "confusion_matrix.csv"
    if cm_path.exists():
        try:
            cm = pd.read_csv(cm_path, index_col=0)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Classification confusion matrix", labels=dict(x="Predicted label", y="Actual label", color="Count"))
            fig_cm.update_layout(template="plotly_white", height=360)
            st.plotly_chart(fig_cm, width="stretch")
        except Exception as exc:
            st.caption(f"Confusion matrix could not be displayed: {exc}")
    cls_model_file = MODEL_DIR / "investment_classifier.joblib"
    if cls_model_file.exists():
        try:
            model_for_importance = joblib.load(cls_model_file)
            tree_model = model_for_importance.named_steps.get("model") if hasattr(model_for_importance, "named_steps") else model_for_importance
            prep = model_for_importance.named_steps.get("preprocessor") if hasattr(model_for_importance, "named_steps") else None
            if hasattr(tree_model, "feature_importances_") and prep is not None:
                names = prep.get_feature_names_out()
                importance = pd.DataFrame({"Feature": names, "Importance": tree_model.feature_importances_}).nlargest(20, "Importance").sort_values("Importance")
                fig_imp = px.bar(importance, x="Importance", y="Feature", orientation="h", title="Top 20 classification feature importances", color="Importance", color_continuous_scale="Teal")
                fig_imp.update_layout(template="plotly_white", height=520, coloraxis_showscale=False)
                st.plotly_chart(fig_imp, width="stretch")
        except Exception as exc:
            st.caption(f"Feature importance is not available for this saved model: {exc}")
    if not cls_path.exists():
        st.warning("Classifier model not found. Run `python src/train_models.py` to create it.")
    else:
        try:
            classifier = joblib.load(cls_path)
        except Exception as exc:
            classifier = None
            st.error(f"Could not load classifier: {exc}")

        with st.container(key="advisor_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                state_opts = options_for("State")
                advisor_state = st.selectbox("State", state_opts, key="adv_state") if state_opts else "Unknown"
                city_opts = sorted(df.loc[df["State"] == advisor_state, "City"].dropna().astype(str).unique().tolist()) if state_opts else options_for("City")
                advisor_city = st.selectbox("City", city_opts, key="adv_city")
                locality_opts = sorted(df.loc[df["City"] == advisor_city, "Locality"].dropna().unique().tolist()) if "Locality" in df else []
                advisor_locality = st.selectbox("Locality", locality_opts) if locality_opts else "Unknown"
                ptype_opts = options_for("Property_Type")
                advisor_type = st.selectbox("Property type", ptype_opts) if ptype_opts else "Unknown"
            with c2:
                advisor_bhk = st.number_input("BHK", 1, 10, 2)
                advisor_size = st.number_input("Size (sq ft)", 100, 20000, 1000, 50)
                advisor_year = st.number_input("Year built", 1950, int(pd.Timestamp.today().year), 2015)
                furnished_opts = options_for("Furnished_Status")
                advisor_furnished = st.selectbox("Furnishing", furnished_opts) if furnished_opts else "Unknown"
            with c3:
                advisor_floor = st.number_input("Floor number", 0, 100, 1)
                advisor_total_floors = st.number_input("Total floors", 1, 100, 5)
                school_count = st.number_input("Nearby schools", 0, 100, 3)
                hospital_count = st.number_input("Nearby hospitals", 0, 100, 2)

            # Additional feature categories are selected from actual dataset values.
            extra_cols = st.columns(3)
            with extra_cols[0]:
                transport_opts = options_for("Public_Transport_Accessibility")
                advisor_transport = st.selectbox("Public transport", transport_opts) if transport_opts else "Unknown"
                parking_opts = options_for("Parking_Space")
                advisor_parking = st.selectbox("Parking", parking_opts) if parking_opts else "Unknown"
            with extra_cols[1]:
                security_opts = options_for("Security")
                advisor_security = st.selectbox("Security", security_opts) if security_opts else "Unknown"
                amenities_opts = options_for("Amenities")
                advisor_amenities = st.selectbox("Amenities", amenities_opts[:100]) if amenities_opts else "Unknown"
            with extra_cols[2]:
                facing_opts = options_for("Facing")
                advisor_facing = st.selectbox("Facing", facing_opts) if facing_opts else "Unknown"
                owner_opts = options_for("Owner_Type")
                advisor_owner = st.selectbox("Owner type", owner_opts) if owner_opts else "Unknown"
                availability_opts = options_for("Availability_Status")
                advisor_availability = st.selectbox("Availability", availability_opts) if availability_opts else "Unknown"

            growth_rate = st.slider("Assumed annual appreciation (%)", 0.0, 15.0, 8.0, 0.5) / 100
            horizon = st.slider("Scenario horizon (years)", 1, 10, 5)
            submitted = st.button("Analyze property", type="primary")

        if submitted:
            input_values = {
                "State": advisor_state, "City": advisor_city, "Locality": advisor_locality,
                "Property_Type": advisor_type, "BHK": advisor_bhk, "Size_in_SqFt": advisor_size,
                "Year_Built": advisor_year, "Furnished_Status": advisor_furnished,
                "Floor_No": advisor_floor, "Total_Floors": advisor_total_floors,
                "Age_of_Property": max(0, pd.Timestamp.today().year - advisor_year),
                "Nearby_Schools": school_count, "Nearby_Hospitals": hospital_count,
                "Public_Transport_Accessibility": advisor_transport, "Parking_Space": advisor_parking,
                "Security": advisor_security, "Amenities": advisor_amenities, "Facing": advisor_facing,
                "Owner_Type": advisor_owner, "Availability_Status": advisor_availability,
                "Amenities_Count": len([x for x in str(advisor_amenities).split(",") if x.strip()]),
                "Has_Parking": int(str(advisor_parking).lower() in ["yes", "1", "true", "available"]),
                "Has_Security": int(str(advisor_security).lower() in ["yes", "1", "true", "gated", "cctv", "guard"]),
            }
            input_frame = pd.DataFrame([input_values])
            try:
                if classifier is None:
                    raise RuntimeError("Classifier model is unavailable.")
                cls_input = feature_row_for_model(classifier, input_frame, df)
                pred = int(classifier.predict(cls_input)[0])
                confidence = float(np.max(classifier.predict_proba(cls_input)[0])) if hasattr(classifier, "predict_proba") else np.nan
                st.metric("Relative investment screening", "Potentially attractive" if pred == 1 else "Needs further review")
                if pd.notna(confidence):
                    st.write(f"Model confidence: **{confidence:.1%}**. This is confidence in the dataset's relative label, not a guarantee of profit.")
                st.caption("The target label was derived from price per sq ft relative to the city median; it does not measure realized investment returns.")
            except Exception as exc:
                st.error(f"The classifier could not analyze these inputs: {exc}")
                st.info("This can happen if the model expects a different feature schema. The rest of the dashboard remains available.")

            if reg_path.exists():
                try:
                    regressor = joblib.load(reg_path)
                    reg_input = feature_row_for_model(regressor, input_frame, df)
                    estimated_price = max(0.0, float(regressor.predict(reg_input)[0]))
                    future_price = estimated_price * ((1 + growth_rate) ** horizon)
                    x1, x2 = st.columns(2)
                    x1.metric("Estimated current price", money_lakh(estimated_price))
                    x2.metric(f"Scenario value in {horizon} years", money_lakh(future_price))
                    st.info(
                        f"Scenario calculation: estimated current value × (1 + {growth_rate:.1%})^{horizon}. "
                        "The dataset contains current listing prices, not observed future prices. "
                        "This is a scenario, not a learned five-year forecast. Review the regression evaluation metrics before relying on its current-price estimate."
                    )
                except Exception as exc:
                    st.warning(f"Current-price regression could not run with these inputs: {exc}")

# -------------------- Page 6: Data quality --------------------
elif page == "Data Quality":
    st.subheader("Dataset quality report")
    original = pd.read_csv(data_path)
    missing_total = int(original.isna().sum().sum())
    duplicate_rows = int(original.duplicated().sum())
    invalid_price = int((pd.to_numeric(original.get("Price_in_Lakhs"), errors="coerce") <= 0).sum())
    invalid_size = int((pd.to_numeric(original.get("Size_in_SqFt"), errors="coerce") <= 0).sum())
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rows", f"{len(original):,}")
    q2.metric("Columns", f"{len(original.columns):,}")
    q3.metric("Missing cells", f"{missing_total:,}")
    q4.metric("Duplicate rows", f"{duplicate_rows:,}")

    st.markdown("#### Missing values by column")
    missing = original.isna().sum().rename("Missing values").reset_index()
    missing.columns = ["Column", "Missing values"]
    missing["Missing %"] = (missing["Missing values"] / max(len(original), 1) * 100).round(2)
    st.dataframe(missing.sort_values("Missing values", ascending=False), width="stretch", hide_index=True)

    st.markdown("#### Basic validation checks")
    checks = pd.DataFrame([
        {"Check": "Price less than or equal to zero", "Rows": invalid_price},
        {"Check": "Area less than or equal to zero", "Rows": invalid_size},
        {"Check": "Duplicate rows", "Rows": duplicate_rows},
    ])
    st.dataframe(checks, width="stretch", hide_index=True)
    if "Price_per_SqFt" in original.columns:
        ppsf = pd.to_numeric(original["Price_per_SqFt"], errors="coerce")
        zero_ppsf = int((ppsf == 0).sum())
        st.metric("Zero price-per-sq-ft values", f"{zero_ppsf:,}")
    st.caption(f"Dashboard data source: {data_path.relative_to(ROOT)}")

st.markdown(
    '<div class="re-footer">'
    '<div class="re-foot-brand">🏡 REAL ESTATE INVESTMENT ADVISOR</div>'
    '<div>Real Estate Investment Advisor | Built with Python, Streamlit, Pandas, Plotly and Scikit-learn</div>'
    '<div class="re-foot-note">Educational decision-support prototype — not financial advice.</div>'
    '</div>',
    unsafe_allow_html=True,
)