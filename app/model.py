# ============================
#  Festival-Aware Dynamic Pricing — Model Logic
#  (Extracted from FastAPI api.py for Streamlit deployment)
# ============================

import os
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from datetime import date

# -----------------------------
# RESOLVE PATHS — works whether launched from project root or app/
# -----------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

_MODEL_PATH    = os.path.join(_ROOT, "pricing_xgb_model.pkl")
_FEATURES_PATH = os.path.join(_ROOT, "feature_list.pkl")
_FESTIVALS_PATH = os.path.join(_ROOT, "festival_dates.pkl")
_CSV_PATH      = os.path.join(_ROOT, "synthetic_pricing_mixed_elasticity.csv")


# -----------------------------
# LOAD ASSETS (cached via Streamlit's @st.cache_resource when called from UI)
# -----------------------------
def load_assets():
    """Load and return (model, FEATURES list, FESTIVAL_DATES dict, dataframe)."""
    model = joblib.load(_MODEL_PATH)
    features = joblib.load(_FEATURES_PATH)
    festival_dates = joblib.load(_FESTIVALS_PATH)
    df = pd.read_csv(_CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return model, features, festival_dates, df


# -----------------------------
# DROPDOWN HELPERS
# -----------------------------
def get_products(df: pd.DataFrame):
    return sorted(df["product_id"].unique().tolist())

def get_categories(df: pd.DataFrame):
    return sorted(df["category"].unique().tolist())

def get_dates(df: pd.DataFrame):
    return sorted(df["date"].dt.strftime("%Y-%m-%d").unique().tolist())


# -----------------------------
# LOOKUP ROW
# -----------------------------
def lookup_row(df: pd.DataFrame, product_id: str, date_str: str) -> pd.Series:
    dt = pd.to_datetime(date_str)
    sub = df[(df["product_id"] == product_id) & (df["date"] == dt)]
    if sub.empty:
        raise ValueError(f"No data found for product '{product_id}' on {date_str}.")
    row = sub.iloc[0].copy()
    # Ensure safe fallback fields
    for col, default in {"lag_1_units": 0, "lag_7_units": 0, "quality_score": 4.5}.items():
        if col not in row:
            row[col] = default
    return row


# -----------------------------
# BUILD FEATURE ROW
# -----------------------------
def build_feature_row(row: pd.Series, date_str: str, FEATURES: list, FESTIVAL_DATES: dict):
    dt = pd.to_datetime(date_str)
    feat = {}

    base_cols = [
        "price", "competitor_price", "discount_pct", "rating",
        "ad_spend", "stock_level", "on_promotion",
        "lag_1_units", "lag_7_units",
    ]
    for col in base_cols:
        feat[col] = float(row[col])

    # Engineered features
    feat["log_price"] = np.log1p(feat["price"])
    feat["log_competitor_price"] = np.log1p(feat["competitor_price"])
    feat["day_of_week"] = dt.weekday()
    feat["is_weekend"] = int(dt.weekday() >= 5)
    feat["month"] = dt.month
    feat["week_of_year"] = int(dt.strftime("%U"))
    feat["seasonality_factor"] = 1.0

    # Festival distances
    festival_distances = {}
    for fest, info in FESTIVAL_DATES.items():
        fest_date = date(dt.year, info["month"], info["day"])
        if fest_date < dt.date():
            fest_date = date(dt.year + 1, info["month"], info["day"])
        days_to = (fest_date - dt.date()).days
        feat[f"fest_{fest}_days_to"] = days_to
        feat[f"fest_{fest}_is_month"] = int(fest_date.month == dt.month)
        festival_distances[fest] = days_to

    feat_series = pd.Series(feat).reindex(FEATURES)
    return feat_series, festival_distances


# -----------------------------
# PRICE OPTIMIZATION
# -----------------------------
def optimize_price(
    model,
    features: pd.Series,
    festival_distances: dict,
    goal: str,
) -> dict:
    base_price = float(features["price"])
    base_demand = float(model.predict(features.values.reshape(1, -1)))
    base_revenue = base_price * base_demand

    def obj(p):
        t = features.copy()
        t["price"] = p
        t["log_price"] = np.log1p(p)
        d = float(model.predict(t.values.reshape(1, -1)))
        if goal == "Revenue":
            return -(p * d)
        elif goal == "Units":
            return -d
        else:  # Profit
            return -((p - 0.6 * p) * d)

    result = minimize_scalar(
        obj,
        bounds=(0.5 * base_price, 1.5 * base_price),
        method="bounded",
    )

    optimal_price = float(result.x)
    optimal_demand = float(model.predict(
        pd.Series({**features.to_dict(), "price": optimal_price,
                   "log_price": np.log1p(optimal_price)}).reindex(features.index).values.reshape(1, -1)
    ))
    optimal_value = float(-result.fun)
    uplift = ((optimal_value / base_revenue) - 1) * 100 if base_revenue != 0 else 0.0

    nearest = min(festival_distances, key=festival_distances.get)
    days = festival_distances[nearest]

    return {
        "optimal_price": optimal_price,
        "optimal_demand": optimal_demand,
        "base_price": base_price,
        "base_demand": base_demand,
        "uplift_percentage": uplift,
        "goal": goal,
        "nearest_festival": nearest,
        "days_to_festival": days,
        "festival_season": days <= 30,
        "all_festival_distances": festival_distances,
    }
