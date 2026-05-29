"""
ML models: price forecasting + anomaly detection
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database import (
    read_prices, save_forecast, save_anomaly,
    read_commodities, read_markets
)


# ─── Feature Engineering ──────────────────────────────────────────────────────

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-series features from a price DataFrame."""
    df = df.copy()
    df["price_date"] = pd.to_datetime(df["price_date"])
    df = df.sort_values("price_date")
    df = df.set_index("price_date").asfreq("D").ffill().reset_index()

    df["day_of_year"]  = df["price_date"].dt.dayofyear
    df["day_of_week"]  = df["price_date"].dt.dayofweek
    df["month"]        = df["price_date"].dt.month
    df["week"]         = df["price_date"].dt.isocalendar().week.astype(int)
    df["sin_doy"]      = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"]      = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["trend"]        = np.arange(len(df))

    # Lag features
    for lag in [1, 3, 7, 14, 30]:
        df[f"lag_{lag}"] = df["modal_price"].shift(lag)

    # Rolling stats
    for win in [7, 14, 30]:
        df[f"roll_mean_{win}"] = df["modal_price"].shift(1).rolling(win).mean()
        df[f"roll_std_{win}"]  = df["modal_price"].shift(1).rolling(win).std()

    df["pct_change_7"]  = df["modal_price"].pct_change(7)
    df["pct_change_30"] = df["modal_price"].pct_change(30)

    return df.dropna()


FEATURE_COLS = [
    "trend", "day_of_year", "day_of_week", "month", "week",
    "sin_doy", "cos_doy",
    "lag_1", "lag_3", "lag_7", "lag_14", "lag_30",
    "roll_mean_7", "roll_mean_14", "roll_mean_30",
    "roll_std_7", "roll_std_14", "roll_std_30",
    "pct_change_7", "pct_change_30",
]


# ─── Forecasting ──────────────────────────────────────────────────────────────

def build_forecast_model(df: pd.DataFrame) -> Tuple:
    """Train a Random Forest model and return (model, scaler, last_features, mape)."""
    feat_df = make_features(df)
    if len(feat_df) < 20:
        return None, None, None, None

    available = [c for c in FEATURE_COLS if c in feat_df.columns]
    X = feat_df[available].values
    y = feat_df["modal_price"].values

    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    if len(X_test) > 0:
        y_pred = model.predict(X_test_s)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    else:
        mape = None

    # Keep last row features for iterative forecasting
    last_feat = feat_df.iloc[-1][available].values
    return model, scaler, last_feat, mape


def forecast_prices(commodity_id: int, market_id: int, horizon: int = 14) -> pd.DataFrame:
    """
    Generate price forecasts for the next `horizon` days.
    Returns DataFrame with columns: forecast_date, predicted_price, lower_bound, upper_bound.
    """
    df = read_prices(commodity_id=commodity_id, market_id=market_id)
    if df.empty or len(df) < 15:
        return pd.DataFrame()

    df = df.sort_values("price_date")
    model, scaler, _, mape = build_forecast_model(df)
    if model is None:
        return pd.DataFrame()

    # Build future feature rows
    last_date = pd.to_datetime(df["price_date"].max())
    history   = df["modal_price"].values
    predictions = []

    extended_df = df.copy()

    for step in range(1, horizon + 1):
        future_date = last_date + timedelta(days=step)
        # Append a synthetic row with NaN price for feature creation
        new_row = extended_df.iloc[-1:].copy()
        new_row["price_date"] = future_date.strftime("%Y-%m-%d")
        new_row["modal_price"] = np.nan
        extended_df = pd.concat([extended_df, new_row], ignore_index=True)

        feat_df = make_features(extended_df.dropna(subset=["modal_price"]))
        if feat_df.empty:
            break

        available = [c for c in FEATURE_COLS if c in feat_df.columns]
        last_feats = feat_df.iloc[-1][available].values.reshape(1, -1)
        last_feats_s = scaler.transform(last_feats)
        pred = model.predict(last_feats_s)[0]
        pred = max(pred, 1.0)

        # Uncertainty grows with horizon
        std_hist = extended_df["modal_price"].dropna().std()
        uncertainty = std_hist * (0.03 * step)
        lower = pred - 1.96 * uncertainty
        upper = pred + 1.96 * uncertainty

        predictions.append({
            "forecast_date": future_date.strftime("%Y-%m-%d"),
            "predicted_price": round(pred, 2),
            "lower_bound": round(max(lower, 1.0), 2),
            "upper_bound": round(upper, 2),
            "model_name": "RandomForest",
            "confidence": round(max(0, 100 - (mape or 10)), 1),
        })

        # Fill back for next iteration's lag features
        extended_df.iloc[-1, extended_df.columns.get_loc("modal_price")] = pred

    result = pd.DataFrame(predictions)
    # Persist to DB
    for _, row in result.iterrows():
        save_forecast(
            commodity_id, market_id,
            row["forecast_date"], row["predicted_price"],
            row["lower_bound"], row["upper_bound"],
            row["model_name"], row["confidence"]
        )
    return result


# ─── Anomaly Detection ────────────────────────────────────────────────────────

def detect_anomalies(commodity_id: int, market_id: int, window: int = 30) -> List[Dict]:
    """
    Multi-method anomaly detection:
    1. Z-score spike detection
    2. IQR fence outliers
    3. Isolation Forest
    Returns list of anomaly dicts.
    """
    df = read_prices(commodity_id=commodity_id, market_id=market_id)
    if df.empty or len(df) < 10:
        return []

    df = df.sort_values("price_date").copy()
    df["price_date"] = pd.to_datetime(df["price_date"])
    prices = df["modal_price"].values
    detected = []

    # ── 1. Z-score ──
    roll_mean = pd.Series(prices).rolling(window, min_periods=5).mean().values
    roll_std  = pd.Series(prices).rolling(window, min_periods=5).std().values
    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = np.where(roll_std > 0, (prices - roll_mean) / roll_std, 0)

    for i, (z, row) in enumerate(zip(z_scores, df.itertuples())):
        if abs(z) > 2.5:
            severity = "Critical" if abs(z) > 4 else "High" if abs(z) > 3 else "Medium"
            direction = "spike" if z > 0 else "drop"
            dev_pct = round(float(z) * float(roll_std[i]) / float(roll_mean[i]) * 100, 2) if roll_mean[i] else 0
            detected.append({
                "price_record_id": None,
                "commodity_id": commodity_id,
                "market_id": market_id,
                "detected_date": str(row.price_date.date()),
                "anomaly_type": f"Z-Score {direction}",
                "severity": severity,
                "deviation_pct": dev_pct,
                "description": f"Price of ₹{row.modal_price:.0f} deviates {abs(z):.1f}σ from {window}-day mean ₹{roll_mean[i]:.0f}",
            })

    # ── 2. IQR fences ──
    q1, q3 = np.percentile(prices, 25), np.percentile(prices, 75)
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 2.5 * iqr, q3 + 2.5 * iqr
    for row in df.itertuples():
        p = row.modal_price
        if p < lower_fence or p > upper_fence:
            dev = (p - np.median(prices)) / np.median(prices) * 100
            detected.append({
                "price_record_id": None,
                "commodity_id": commodity_id,
                "market_id": market_id,
                "detected_date": str(row.price_date.date()),
                "anomaly_type": "IQR Outlier",
                "severity": "High" if abs(dev) > 50 else "Medium",
                "deviation_pct": round(dev, 2),
                "description": f"Price ₹{p:.0f} outside IQR fence [{lower_fence:.0f}, {upper_fence:.0f}]",
            })

    # ── 3. Isolation Forest ──
    if len(prices) >= 20:
        feat_matrix = np.column_stack([
            prices,
            np.abs(np.diff(prices, prepend=prices[0])),
        ])
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(feat_matrix)
        scores = iso.score_samples(feat_matrix)
        for i, (pred, score, row) in enumerate(zip(preds, scores, df.itertuples())):
            if pred == -1:
                detected.append({
                    "price_record_id": None,
                    "commodity_id": commodity_id,
                    "market_id": market_id,
                    "detected_date": str(row.price_date.date()),
                    "anomaly_type": "Isolation Forest",
                    "severity": "Medium",
                    "deviation_pct": round(abs(score) * 100, 2),
                    "description": f"Anomaly detected by Isolation Forest (score={score:.3f}). Price: ₹{row.modal_price:.0f}",
                })

    # Deduplicate by date + type
    seen = set()
    unique = []
    for a in detected:
        key = (a["detected_date"], a["anomaly_type"])
        if key not in seen:
            seen.add(key)
            unique.append(a)

    # Persist
    for a in unique:
        save_anomaly(**a)

    return unique


def run_anomaly_detection_all():
    """Run anomaly detection for all commodity-market pairs."""
    commodities = read_commodities()
    markets     = read_markets()
    total = 0
    for _, c in commodities.iterrows():
        for _, m in markets.iterrows():
            anomalies = detect_anomalies(c["id"], m["id"])
            total += len(anomalies)
    return total


def run_forecast_all(horizon: int = 14):
    """Run forecasts for all commodity-market pairs."""
    commodities = read_commodities()
    markets     = read_markets()
    total = 0
    for _, c in commodities.iterrows():
        for _, m in markets.iterrows():
            result = forecast_prices(c["id"], m["id"], horizon)
            total += len(result)
    return total
