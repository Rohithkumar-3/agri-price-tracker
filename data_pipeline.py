"""
Data pipeline: fetch, clean, and store commodity price data.
Uses India's data.gov.in / Agmarknet API.
Falls back to synthetic realistic data when the API is unreachable.
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta
from typing import Tuple, List, Dict
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import (
    create_commodity, create_market, upsert_price_record,
    read_commodities, read_markets, log_scheduler
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── API Config ───────────────────────────────────────────────────────────────

AGMARKNET_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
AGMARKNET_BASE_URL    = "https://api.data.gov.in/resource"

# Field name normalization — API returns inconsistent casing
FIELD_MAP = {
    "min_price":   ["min_price",   "Min Price",   "min price",   "Min_Price"],
    "max_price":   ["max_price",   "Max Price",   "max price",   "Max_Price"],
    "modal_price": ["modal_price", "Modal Price", "modal price", "Modal_Price", "modal_Price"],
    "arrivals":    ["arrivals",    "Arrivals",    "arrival",     "Arrival"],
    "commodity":   ["commodity",   "Commodity"],
    "market":      ["market",      "Market",      "market_name", "Market_Name"],
    "state":       ["state",       "State"],
    "district":    ["district",    "District"],
    "date":        ["arrival_date","Arrival_Date","date",        "Date"],
}

# ─── Commodity & Market Seed Data ─────────────────────────────────────────────

COMMODITY_CATALOG = {
    "Cereals":    ["Rice", "Wheat", "Maize", "Barley", "Jowar", "Bajra", "Ragi"],
    "Pulses":     ["Tur Dal", "Moong Dal", "Urad Dal", "Chana Dal", "Masoor Dal"],
    "Vegetables": ["Tomato", "Onion", "Potato", "Brinjal", "Cabbage", "Cauliflower",
                   "Carrot", "Spinach", "Bitter Gourd", "Capsicum"],
    "Fruits":     ["Banana", "Mango", "Apple", "Grapes", "Orange", "Papaya"],
    "Spices":     ["Turmeric", "Chilli", "Coriander", "Cumin", "Ginger", "Garlic"],
    "Oilseeds":   ["Groundnut", "Mustard", "Soybean", "Sunflower", "Sesame"],
}

MARKET_CATALOG = [
    ("Azadpur",            "Delhi",          "North West Delhi"),
    ("Vashi",              "Maharashtra",    "Navi Mumbai"),
    ("Bowenpally",         "Telangana",      "Hyderabad"),
    ("Koyambedu",          "Tamil Nadu",     "Chennai"),
    ("Yeshwanthpur",       "Karnataka",      "Bengaluru"),
    ("Gultekdi",           "Maharashtra",    "Pune"),
    ("Ahmedabad Main",     "Gujarat",        "Ahmedabad"),
    ("Kolkata Main",       "West Bengal",    "Kolkata"),
    ("Patna Main",         "Bihar",          "Patna"),
    ("Lucknow Main",       "Uttar Pradesh",  "Lucknow"),
    ("Jaipur Sabzi Mandi", "Rajasthan",      "Jaipur"),
    ("Bhopal Main",        "Madhya Pradesh", "Bhopal"),
]

BASE_PRICES = {
    "Rice": 2200, "Wheat": 2100, "Maize": 1800, "Barley": 1700,
    "Jowar": 2000, "Bajra": 1900, "Ragi": 2300,
    "Tur Dal": 8500, "Moong Dal": 9500, "Urad Dal": 9000,
    "Chana Dal": 7000, "Masoor Dal": 7500,
    "Tomato": 1500, "Onion": 1800, "Potato": 1400, "Brinjal": 1200,
    "Cabbage": 900, "Cauliflower": 1100, "Carrot": 1600, "Spinach": 1000,
    "Bitter Gourd": 1800, "Capsicum": 2200,
    "Banana": 2000, "Mango": 5000, "Apple": 8000, "Grapes": 4500,
    "Orange": 3500, "Papaya": 1800,
    "Turmeric": 8000, "Chilli": 12000, "Coriander": 7000,
    "Cumin": 18000, "Ginger": 6000, "Garlic": 12000,
    "Groundnut": 5500, "Mustard": 5200, "Soybean": 4800,
    "Sunflower": 5000, "Sesame": 9000,
}

# ─── API Key Loader ───────────────────────────────────────────────────────────

def _get_api_key() -> str:
    """
    Load Agmarknet API key from (priority order):
      1. AGMARKNET_API_KEY environment variable
      2. .env file in project root
      3. config.txt file in project root
    """
    # 1. Environment variable
    key = os.environ.get("AGMARKNET_API_KEY", "").strip()
    if key and key != "your_key_here":
        return key

    # 2. .env file
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("AGMARKNET_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and val != "your_key_here":
                        return val

    # 3. config.txt (plain key only)
    cfg_path = os.path.join(os.path.dirname(__file__), "config.txt")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            val = f.read().strip()
            if val:
                return val

    return ""


def _extract_field(record: dict, field: str):
    """Try all known key variants for a field."""
    for key in FIELD_MAP.get(field, [field]):
        if key in record:
            return record[key]
    return None

# ─── Reference Data Seeding ───────────────────────────────────────────────────

def seed_reference_data():
    logger.info("Seeding reference data...")
    for category, commodities in COMMODITY_CATALOG.items():
        for name in commodities:
            create_commodity(name, category)
    for market_name, state, district in MARKET_CATALOG:
        create_market(market_name, state, district)
    logger.info("Reference data seeded.")

# ─── Live API Fetch ───────────────────────────────────────────────────────────

def try_fetch_agmarknet(commodity: str, state: str, price_date: str) -> List[Dict]:
    """
    Fetch from data.gov.in Agmarknet API.
    Returns list of normalised record dicts, or [] if unavailable / no key.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.debug("No Agmarknet API key — using synthetic data.")
        return []

    url = f"{AGMARKNET_BASE_URL}/{AGMARKNET_RESOURCE_ID}"
    params = {
        "api-key":              api_key,
        "format":               "json",
        "limit":                100,
        "filters[commodity]":   commodity,
        "filters[state]":       state,
        "filters[arrival_date]": price_date,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 401:
            logger.warning("Agmarknet: invalid/expired API key.")
            return []
        if resp.status_code != 200:
            logger.debug(f"Agmarknet HTTP {resp.status_code} for {commodity}/{state}/{price_date}")
            return []

        raw = resp.json()
        if raw.get("status") == "error":
            logger.warning(f"Agmarknet API error: {raw.get('message')}")
            return []

        records = raw.get("records", [])
        cleaned = []
        for rec in records:
            cleaned.append({
                "min_price":   _extract_field(rec, "min_price"),
                "max_price":   _extract_field(rec, "max_price"),
                "modal_price": _extract_field(rec, "modal_price"),
                "arrivals":    _extract_field(rec, "arrivals") or 0,
            })
        return [r for r in cleaned if r["modal_price"] is not None]

    except requests.exceptions.Timeout:
        logger.warning(f"Agmarknet timeout: {commodity}/{state}")
    except Exception as e:
        logger.debug(f"Agmarknet error: {e}")
    return []

# ─── Synthetic Price Generator ────────────────────────────────────────────────

def generate_synthetic_price(commodity: str, market: str, price_date: date,
                              seed_offset: int = 0) -> Dict:
    base = BASE_PRICES.get(commodity, 2000)
    day_of_year = price_date.timetuple().tm_yday
    seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
    rng = np.random.default_rng(
        int(price_date.strftime("%Y%m%d")) + hash(commodity + market) % 10000 + seed_offset
    )
    noise = rng.normal(0, 0.03)
    market_premium = {
        "Azadpur": 1.05, "Vashi": 1.08, "Bowenpally": 0.97,
        "Koyambedu": 1.02, "Yeshwanthpur": 1.06, "Gultekdi": 1.04,
    }.get(market, 1.0)
    modal_price = round(base * seasonal * market_premium * (1 + noise), 2)
    spread = rng.uniform(0.03, 0.08)
    return {
        "min_price":   round(modal_price * (1 - spread), 2),
        "max_price":   round(modal_price * (1 + spread), 2),
        "modal_price": modal_price,
        "arrivals":    round(rng.uniform(50, 500), 1),
    }

# ─── Data Cleaning ────────────────────────────────────────────────────────────

def clean_price_record(record: Dict):
    try:
        min_p  = float(str(record.get("min_price",   0)).replace(",", ""))
        max_p  = float(str(record.get("max_price",   0)).replace(",", ""))
        modal  = float(str(record.get("modal_price", 0)).replace(",", ""))
        if min_p <= 0 or max_p <= 0 or modal <= 0:
            return None
        if min_p > modal or modal > max_p:
            modal = (min_p + max_p) / 2
        if max_p - min_p > modal * 2:
            return None
        record.update(min_price=min_p, max_price=max_p, modal_price=modal)
        return record
    except (ValueError, TypeError):
        return None

# ─── Main ETL ─────────────────────────────────────────────────────────────────

def fetch_and_store_prices(target_date: date = None, days_back: int = 30) -> int:
    if target_date is None:
        target_date = date.today()

    commodities_df = read_commodities()
    markets_df     = read_markets()

    if commodities_df.empty:
        seed_reference_data()
        commodities_df = read_commodities()
        markets_df     = read_markets()

    commodity_map = dict(zip(commodities_df["name"], commodities_df["id"]))
    market_map    = {(r["market_name"], r["state"]): r["id"] for _, r in markets_df.iterrows()}
    dates_to_process = [target_date - timedelta(days=i) for i in range(days_back)]

    total_inserted = 0
    for proc_date in dates_to_process:
        date_str = proc_date.strftime("%Y-%m-%d")
        for commodity_name, commodity_id in commodity_map.items():
            for (market_name, state), market_id in market_map.items():
                live = try_fetch_agmarknet(commodity_name, state, date_str)
                inserted_live = False
                for rec in live:
                    cleaned = clean_price_record(rec)
                    if cleaned:
                        upsert_price_record(
                            commodity_id, market_id, date_str,
                            cleaned["min_price"], cleaned["max_price"],
                            cleaned["modal_price"], cleaned["arrivals"], "Agmarknet"
                        )
                        total_inserted += 1
                        inserted_live = True

                if not inserted_live:
                    synth = generate_synthetic_price(commodity_name, market_name, proc_date)
                    upsert_price_record(
                        commodity_id, market_id, date_str,
                        synth["min_price"], synth["max_price"],
                        synth["modal_price"], synth["arrivals"], "Synthetic"
                    )
                    total_inserted += 1

    logger.info(f"ETL complete — {total_inserted} records processed.")
    return total_inserted


def run_initial_load():
    logger.info("Running initial historical load (180 days)...")
    seed_reference_data()
    count = fetch_and_store_prices(days_back=180)
    log_scheduler("initial_load", "success", count)
    return count


if __name__ == "__main__":
    from database import init_db
    init_db()
    run_initial_load()
