"""
FastAPI REST API for Commodity Market Platform
Run with: uvicorn api:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import date
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database import (
    init_db, read_prices, read_commodities, read_markets,
    read_anomalies, read_forecasts, read_insights, get_summary_stats,
    get_state_analytics, get_top_commodities_by_volatility,
    create_commodity, update_commodity, delete_commodity,
    create_market, delete_market, upsert_price_record, delete_price_record,
    read_scheduler_log
)
from data_pipeline import fetch_and_store_prices, seed_reference_data
from ml_models import forecast_prices, detect_anomalies
from ai_insights import generate_price_movement_insight, bulk_generate_insights

app = FastAPI(
    title="Commodity Market Intelligence API",
    description="Real-time agricultural commodity price analytics for Indian markets",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class CommodityCreate(BaseModel):
    name: str
    category: str
    unit: str = "Quintal"

class CommodityUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    unit: Optional[str] = None

class MarketCreate(BaseModel):
    market_name: str
    state: str
    district: str

class PriceCreate(BaseModel):
    commodity_id: int
    market_id: int
    price_date: str
    min_price: float
    max_price: float
    modal_price: float
    arrivals: float = 0.0
    source: str = "Manual"


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "Commodity Market API"}


# ─── Summary ─────────────────────────────────────────────────────────────────

@app.get("/api/summary")
def summary():
    return get_summary_stats()

@app.get("/api/analytics/state")
def state_analytics():
    df = get_state_analytics()
    return df.to_dict(orient="records")

@app.get("/api/analytics/volatility")
def volatility(limit: int = 10):
    df = get_top_commodities_by_volatility(limit)
    return df.to_dict(orient="records")


# ─── Commodities CRUD ─────────────────────────────────────────────────────────

@app.get("/api/commodities")
def list_commodities():
    return read_commodities().to_dict(orient="records")

@app.post("/api/commodities", status_code=201)
def add_commodity(body: CommodityCreate):
    cid = create_commodity(body.name, body.category, body.unit)
    return {"id": cid, "message": "Commodity created"}

@app.put("/api/commodities/{cid}")
def modify_commodity(cid: int, body: CommodityUpdate):
    update_commodity(cid, body.name, body.category, body.unit)
    return {"message": "Updated"}

@app.delete("/api/commodities/{cid}")
def remove_commodity(cid: int):
    delete_commodity(cid)
    return {"message": "Deleted"}


# ─── Markets CRUD ─────────────────────────────────────────────────────────────

@app.get("/api/markets")
def list_markets(state: Optional[str] = None):
    df = read_markets()
    if state:
        df = df[df["state"].str.lower() == state.lower()]
    return df.to_dict(orient="records")

@app.post("/api/markets", status_code=201)
def add_market(body: MarketCreate):
    mid = create_market(body.market_name, body.state, body.district)
    return {"id": mid, "message": "Market created"}

@app.delete("/api/markets/{mid}")
def remove_market(mid: int):
    delete_market(mid)
    return {"message": "Deleted"}


# ─── Prices CRUD ─────────────────────────────────────────────────────────────

@app.get("/api/prices")
def list_prices(
    commodity_id: Optional[int] = None,
    market_id:    Optional[int] = None,
    start_date:   Optional[str] = None,
    end_date:     Optional[str] = None,
    limit:        int = 500,
):
    df = read_prices(commodity_id, market_id, start_date, end_date)
    return df.head(limit).to_dict(orient="records")

@app.post("/api/prices", status_code=201)
def add_price(body: PriceCreate):
    upsert_price_record(
        body.commodity_id, body.market_id, body.price_date,
        body.min_price, body.max_price, body.modal_price,
        body.arrivals, body.source
    )
    return {"message": "Price record upserted"}

@app.delete("/api/prices/{record_id}")
def remove_price(record_id: int):
    delete_price_record(record_id)
    return {"message": "Deleted"}


# ─── Forecasts ────────────────────────────────────────────────────────────────

@app.get("/api/forecast")
def get_forecast(commodity_id: int, market_id: int, horizon: int = 14):
    df = forecast_prices(commodity_id, market_id, horizon)
    if df.empty:
        raise HTTPException(404, "Insufficient data for forecasting")
    return df.to_dict(orient="records")


# ─── Anomalies ────────────────────────────────────────────────────────────────

@app.get("/api/anomalies")
def list_anomalies(days: int = 30):
    return read_anomalies(days).to_dict(orient="records")

@app.post("/api/anomalies/detect")
def trigger_anomaly_detection(commodity_id: int, market_id: int, background: BackgroundTasks):
    background.add_task(detect_anomalies, commodity_id, market_id)
    return {"message": "Anomaly detection started in background"}


# ─── AI Insights ─────────────────────────────────────────────────────────────

@app.get("/api/insights")
def list_insights(days: int = 7):
    return read_insights(days).to_dict(orient="records")

@app.post("/api/insights/generate")
def trigger_insights(commodity_id: int, market_id: int):
    text = generate_price_movement_insight(commodity_id, market_id)
    return {"insight": text}


# ─── ETL / Scheduler ─────────────────────────────────────────────────────────

@app.post("/api/etl/run")
def run_etl(days_back: int = 1, background: BackgroundTasks = None):
    if background:
        background.add_task(fetch_and_store_prices, None, days_back)
        return {"message": f"ETL started for {days_back} days"}
    count = fetch_and_store_prices(days_back=days_back)
    return {"records_processed": count}

@app.get("/api/scheduler/log")
def scheduler_log(limit: int = 50):
    return read_scheduler_log(limit).to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
