# ============================
#  Festival-Aware Dynamic Pricing API
#  (FastAPI + CSV lookup + safe feature rebuild)
# ============================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from datetime import date


# -----------------------------
# LOAD MODEL + DATASET
# -----------------------------
model = joblib.load("pricing_xgb_model.pkl")
FEATURES = joblib.load("feature_list.pkl")
FESTIVAL_DATES = joblib.load("festival_dates.pkl")

df = pd.read_csv("synthetic_pricing_mixed_elasticity.csv")
df["date"] = pd.to_datetime(df["date"])

app = FastAPI(title="Dynamic Pricing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# INPUT FORMAT
# -----------------------------
class PricingRequest(BaseModel):
    product_id: str
    category: str
    date: str
    goal: str = "Revenue"


# -----------------------------
# LOOKUP ROW
# -----------------------------
def lookup_row(product_id, date_str):
    dt = pd.to_datetime(date_str)

    sub = df[(df["product_id"] == product_id) & (df["date"] == dt)]
    if sub.empty:
        raise HTTPException(404, f"No row found for {product_id} on {date_str}")

    row = sub.iloc[0]

    # Ensure missing fields exist
    safe_fields = {
        "lag_1_units": 0,
        "lag_7_units": 0,
        "quality_score": 4.5,
    }
    for col, default_value in safe_fields.items():
        if col not in row:
            row[col] = default_value

    return row


# -----------------------------
# BUILD FEATURES
# -----------------------------
def build_feature_row(row, date_str):
    dt = pd.to_datetime(date_str)
    feat = {}

    base_cols = [
        "price", "competitor_price", "discount_pct", "rating",
        "ad_spend", "stock_level", "on_promotion",
        "lag_1_units", "lag_7_units"
    ]

    for col in base_cols:
        feat[col] = float(row[col])

    # Engineered Features
    feat["log_price"] = np.log1p(feat["price"])
    feat["log_competitor_price"] = np.log1p(feat["competitor_price"])

    feat["day_of_week"] = dt.weekday()
    feat["is_weekend"] = int(dt.weekday() >= 5)
    feat["month"] = dt.month
    feat["week_of_year"] = int(dt.strftime("%U"))
    feat["seasonality_factor"] = 1.0

    # Festival Features
    festival_distances = {}

    for fest, info in FESTIVAL_DATES.items():
        fest_date = date(dt.year, info["month"], info["day"])
        if fest_date < dt.date():
            fest_date = date(dt.year + 1, info["month"], info["day"])

        days_to = (fest_date - dt.date()).days

        feat[f"fest_{fest}_days_to"] = days_to
        feat[f"fest_{fest}_is_month"] = int(fest_date.month == dt.month)

        festival_distances[fest] = days_to

    # Match training feature order
    feat_series = pd.Series(feat).reindex(FEATURES)

    return feat_series, festival_distances


# -----------------------------
# PRICE OPTIMIZATION
# -----------------------------
def optimize_single_row(features, goal):
    base_price = float(features["price"])
    base_demand = float(model.predict(features.values.reshape(1, -1)))

    base_cost = 0.6 * base_price
    base_revenue = base_price * base_demand
    base_profit = (base_price - base_cost) * base_demand

    def obj(p):
        t = features.copy()
        t["price"] = p
        t["log_price"] = np.log1p(p)
        d = float(model.predict(t.values.reshape(1, -1)))

        if goal == "Revenue":
            return -(p * d)
        elif goal == "Units":
            return -d
        else:
            return -((p - 0.6 * p) * d)

    result = minimize_scalar(obj, bounds=(0.5 * base_price, 1.5 * base_price), method="bounded")

    return {
        "optimal_price": float(result.x),
        "optimal_value": float(-result.fun),
        "base_price": base_price,
        "base_demand": base_demand,
        "uplift_percentage": float((( -result.fun / base_revenue ) - 1) * 100),
    }

# -----------------------------
# DROPDOWN ENDPOINTS
# -----------------------------
@app.get("/get_products")
def get_products():
    return {"products": sorted(df["product_id"].unique().tolist())}


@app.get("/get_categories")
def get_categories():
    return {"categories": sorted(df["category"].unique().tolist())}


@app.get("/get_dates")
def get_dates():
    dates = sorted(df["date"].dt.strftime("%Y-%m-%d").unique().tolist())
    return {"dates": dates}


# -----------------------------
# MAIN ENDPOINT
# -----------------------------
@app.post("/optimize_price")
def optimize_price(req: PricingRequest):

    row = lookup_row(req.product_id, req.date)
    features, dists = build_feature_row(row, req.date)
    result = optimize_single_row(features, req.goal)

    nearest = min(dists, key=dists.get)
    days = dists[nearest]

    summary = {
        "nearest_festival": nearest,
        "days_to_festival": days,
        "festival_season": days <= 30
    }

    return {
        "product_id": req.product_id,
        "date": req.date,
        "goal": req.goal,
        "result": result,
        "festival_summary": summary
    }
