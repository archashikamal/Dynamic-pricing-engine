# ============================
#  Dynamic Pricing — Streamlit App
# ============================

import sys
import os

# Ensure project root is on sys.path so `app.model` resolves correctly
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_APP_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import streamlit as st
from app.model import (
    load_assets,
    get_products,
    get_categories,
    get_dates,
    lookup_row,
    build_feature_row,
    optimize_price,
)

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Pricing Engine",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
}
div[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* ── Selectbox ── */
div[data-baseweb="select"] {
    border-radius: 10px !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 20px;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    box-shadow: 0 6px 28px rgba(99,102,241,0.6);
    transform: translateY(-1px);
}

/* ── Info/Success/Error boxes ── */
.info-card {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 14px;
    padding: 18px 22px;
    margin-top: 8px;
}
.festival-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f59e0b, #ef4444);
    color: white;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-left: 8px;
    vertical-align: middle;
}
.header-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a5b4fc, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
}
.header-sub {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 4px;
}
.section-header {
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 28px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}
.stMarkdown p { color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD ASSETS (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading pricing model…")
def load():
    return load_assets()

model, FEATURES, FESTIVAL_DATES, df = load()


# ──────────────────────────────────────────────
# SIDEBAR — INPUTS
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    products   = get_products(df)
    categories = get_categories(df)
    dates      = get_dates(df)

    product_id = st.selectbox("🏷️ Product ID",  products,  index=0)
    category   = st.selectbox("📦 Category",     categories, index=0)
    date_str   = st.selectbox("📅 Date",         dates,      index=0)
    goal       = st.selectbox(
        "🎯 Optimization Goal",
        ["Revenue", "Units", "Profit"],
        help="What metric should the optimizer maximize?",
    )

    st.markdown("---")
    run_btn = st.button("💹 Get Optimal Price", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b'>Powered by XGBoost + SciPy<br>Festival-aware pricing engine</small>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# MAIN — HEADER
# ──────────────────────────────────────────────
st.markdown('<p class="header-title">💹 Dynamic Pricing Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="header-sub">Festival-aware price optimization using XGBoost + SciPy</p>', unsafe_allow_html=True)
st.markdown("")

# ──────────────────────────────────────────────
# MAIN — RUN OPTIMIZATION
# ──────────────────────────────────────────────
if run_btn:
    with st.spinner("Optimizing price…"):
        try:
            row          = lookup_row(df, product_id, date_str)
            features, fd = build_feature_row(row, date_str, FEATURES, FESTIVAL_DATES)
            result       = optimize_price(model, features, fd, goal)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    # ── Key Metrics ──
    st.markdown('<p class="section-header">📊 Pricing Results</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Optimal Price",
        f"₹{result['optimal_price']:.2f}",
        delta=f"{result['uplift_percentage']:+.2f}% vs base",
        delta_color="normal",
    )
    c2.metric("Base Price",      f"₹{result['base_price']:.2f}")
    c3.metric("Predicted Demand",f"{result['optimal_demand']:.1f} units")
    c4.metric("Uplift",          f"{result['uplift_percentage']:.2f}%",
              delta=f"Goal: {result['goal']}")

    # ── Festival Insight ──
    st.markdown('<p class="section-header">🎉 Festival Insight</p>', unsafe_allow_html=True)

    fest_name = result["nearest_festival"].replace("_", " ").title()
    days      = result["days_to_festival"]
    in_season = result["festival_season"]

    badge_html = '<span class="festival-badge">🔥 PEAK SEASON</span>' if in_season else ""
    st.markdown(
        f"""<div class="info-card">
            <b style="color:#e2e8f0; font-size:1.1rem">{fest_name}</b> {badge_html}<br>
            <span style="color:#94a3b8; font-size:0.9rem">
                {'🎊 Festival is within 30 days — pricing boosted for peak demand!' if in_season
                 else f'📅 {days} day{"s" if days != 1 else ""} away — planning ahead.'}
            </span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── All Festival Distances ──
    with st.expander("📆 All Festival Distances"):
        fest_df = (
            pd.DataFrame(
                [(k.replace("_", " ").title(), v) for k, v in result["all_festival_distances"].items()],
                columns=["Festival", "Days Away"],
            )
            .sort_values("Days Away")
            .reset_index(drop=True)
        )
        st.dataframe(fest_df, use_container_width=True, hide_index=True)

    # ── Raw row snapshot ──
    with st.expander("🔍 Raw Product Row (from dataset)"):
        st.dataframe(row.to_frame().T, use_container_width=True)

else:
    # ── Welcome placeholder ──
    st.markdown(
        """<div class="info-card" style="text-align:center; padding:48px 32px;">
            <div style="font-size:3rem; margin-bottom:16px;">💹</div>
            <h3 style="color:#e2e8f0; margin-bottom:8px;">Ready to optimize pricing</h3>
            <p style="color:#94a3b8;">
                Select a <b>Product</b>, <b>Category</b>, <b>Date</b>, and <b>Goal</b> from the sidebar,
                then click <b>Get Optimal Price</b>.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )
