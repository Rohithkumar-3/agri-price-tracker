"""
database.py — File-based storage (JSON + CSV).
No SQLite. No C extensions. Works on Python 3.10-3.14, Streamlit Cloud, everywhere.
"""
import os, json, csv
import pandas as pd
from datetime import date, timedelta
from typing import Optional, Dict, Any

# ── Storage path ──────────────────────────────────────────────────────────────
def _base() -> str:
    src = os.path.dirname(os.path.abspath(__file__))
    p = "/tmp/agri_data" if not os.access(src, os.W_OK) else os.path.join(src, "data")
    os.makedirs(p, exist_ok=True)
    return p

def _p(name: str) -> str:
    return os.path.join(_base(), name)

def _rj(name: str, default):
    f = _p(name)
    if not os.path.exists(f): return default
    with open(f) as fh: return json.load(fh)

def _wj(name: str, data):
    with open(_p(name), "w") as fh: json.dump(data, fh, default=str)

def _rcsv(name: str) -> pd.DataFrame:
    f = _p(name)
    if not os.path.exists(f) or os.path.getsize(f) == 0: return pd.DataFrame()
    return pd.read_csv(f, dtype=str)

def _wcsv(name: str, df: pd.DataFrame):
    df.to_csv(_p(name), index=False)

# ── Init (no-op for file storage) ────────────────────────────────────────────
def init_db():
    _base()  # just ensure directory exists

# ── App meta ──────────────────────────────────────────────────────────────────
def get_meta(key: str) -> Optional[str]:
    return _rj("meta.json", {}).get(key)

def set_meta(key: str, value: str):
    m = _rj("meta.json", {})
    m[key] = value
    _wj("meta.json", m)

# ── Commodities ───────────────────────────────────────────────────────────────
def create_commodity(name: str, category: str, unit: str = "Quintal") -> int:
    d = _rj("commodities.json", {})
    if name not in d:
        d[name] = {"id": max((v["id"] for v in d.values()), default=0)+1,
                   "name": name, "category": category, "unit": unit}
        _wj("commodities.json", d)
    return d[name]["id"]

def read_commodities() -> pd.DataFrame:
    d = _rj("commodities.json", {})
    if not d: return pd.DataFrame(columns=["id","name","category","unit"])
    return pd.DataFrame(list(d.values())).sort_values("name").reset_index(drop=True)

def update_commodity(cid, name=None, category=None, unit=None):
    d = _rj("commodities.json", {})
    for k,v in d.items():
        if v["id"]==cid:
            if name: d[k]["name"]=name
            if category: d[k]["category"]=category
            if unit: d[k]["unit"]=unit
    _wj("commodities.json", d)

def delete_commodity(cid):
    d = _rj("commodities.json", {})
    _wj("commodities.json", {k:v for k,v in d.items() if v["id"]!=cid})

# ── Markets ───────────────────────────────────────────────────────────────────
def create_market(market_name: str, state: str, district: str) -> int:
    d = _rj("markets.json", {})
    k = f"{market_name}|{state}"
    if k not in d:
        d[k] = {"id": max((v["id"] for v in d.values()), default=0)+1,
                 "market_name": market_name, "state": state, "district": district}
        _wj("markets.json", d)
    return d[k]["id"]

def read_markets() -> pd.DataFrame:
    d = _rj("markets.json", {})
    if not d: return pd.DataFrame(columns=["id","market_name","state","district"])
    return pd.DataFrame(list(d.values())).sort_values(["state","market_name"]).reset_index(drop=True)

def delete_market(mid):
    d = _rj("markets.json", {})
    _wj("markets.json", {k:v for k,v in d.items() if v["id"]!=mid})

# ── Price records ─────────────────────────────────────────────────────────────
_PC = ["commodity_id","market_id","price_date","min_price","max_price","modal_price","arrivals","source"]

def _lp() -> pd.DataFrame:
    df = _rcsv("prices.csv")
    if df.empty: return pd.DataFrame(columns=_PC)
    for c in ["min_price","max_price","modal_price","arrivals"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["commodity_id"] = df["commodity_id"].astype(int)
    df["market_id"]    = df["market_id"].astype(int)
    return df

def upsert_price_record(commodity_id, market_id, price_date,
                        min_price, max_price, modal_price, arrivals=0, source="Synthetic"):
    df   = _lp()
    row  = {"commodity_id":int(commodity_id),"market_id":int(market_id),
            "price_date":str(price_date),"min_price":float(min_price),
            "max_price":float(max_price),"modal_price":float(modal_price),
            "arrivals":float(arrivals),"source":source}
    mask = ((df["commodity_id"]==int(commodity_id))&
            (df["market_id"]==int(market_id))&
            (df["price_date"]==str(price_date)))
    if mask.any():
        for k,v in row.items(): df.loc[mask,k]=v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _wcsv("prices.csv", df)

def bulk_insert_prices(records: list):
    if not records: return
    new = pd.DataFrame([
        {"commodity_id":int(r[0]),"market_id":int(r[1]),"price_date":str(r[2]),
         "min_price":float(r[3]),"max_price":float(r[4]),"modal_price":float(r[5]),
         "arrivals":float(r[6]),"source":str(r[7])} for r in records])
    ex = _lp()
    if ex.empty:
        _wcsv("prices.csv", new)
    else:
        combined = pd.concat([ex, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["commodity_id","market_id","price_date"], keep="last")
        _wcsv("prices.csv", combined)

def _join_prices(df):
    if df.empty: return df
    c = read_commodities()
    m = read_markets()
    df = df.merge(c[["id","name","category","unit"]].rename(columns={"id":"commodity_id","name":"commodity_name"}), on="commodity_id", how="left")
    df = df.merge(m[["id","market_name","state","district"]].rename(columns={"id":"market_id"}), on="market_id", how="left")
    return df

def read_prices(commodity_id=None, market_id=None, start_date=None, end_date=None) -> pd.DataFrame:
    df = _lp()
    if df.empty: return df
    if commodity_id: df=df[df["commodity_id"]==int(commodity_id)]
    if market_id:    df=df[df["market_id"]==int(market_id)]
    if start_date:   df=df[df["price_date"]>=str(start_date)]
    if end_date:     df=df[df["price_date"]<=str(end_date)]
    return _join_prices(df).sort_values("price_date", ascending=False).reset_index(drop=True)

def delete_price_record(record_id): pass

# ── Anomalies ─────────────────────────────────────────────────────────────────
def save_anomaly(commodity_id, market_id, detected_date, anomaly_type,
                 severity, deviation_pct, description, price_record_id=None):
    d = _rj("anomalies.json", [])
    d.append({"commodity_id":int(commodity_id),"market_id":int(market_id),
              "detected_date":str(detected_date),"anomaly_type":anomaly_type,
              "severity":severity,"deviation_pct":float(deviation_pct or 0),
              "description":description})
    _wj("anomalies.json", d[-500:])

def read_anomalies(days=30) -> pd.DataFrame:
    d = _rj("anomalies.json", [])
    if not d: return pd.DataFrame()
    df = pd.DataFrame(d)
    df = df[df["detected_date"]>=str(date.today()-timedelta(days=days))]
    if df.empty: return df
    c = read_commodities(); m = read_markets()
    df = df.merge(c[["id","name"]].rename(columns={"id":"commodity_id","name":"commodity_name"}), on="commodity_id", how="left")
    df = df.merge(m[["id","market_name","state"]].rename(columns={"id":"market_id"}), on="market_id", how="left")
    return df.sort_values("detected_date", ascending=False).reset_index(drop=True)

# ── Forecasts ─────────────────────────────────────────────────────────────────
def save_forecast(commodity_id, market_id, forecast_date, predicted_price,
                  lower_bound=None, upper_bound=None, model_name="RandomForest", confidence=None):
    d = _rj("forecasts.json", [])
    k = f"{commodity_id}_{market_id}_{forecast_date}"
    d = [x for x in d if f"{x['commodity_id']}_{x['market_id']}_{x['forecast_date']}"!=k]
    d.append({"commodity_id":int(commodity_id),"market_id":int(market_id),
              "forecast_date":str(forecast_date),"predicted_price":float(predicted_price),
              "lower_bound":float(lower_bound or predicted_price*0.95),
              "upper_bound":float(upper_bound or predicted_price*1.05),
              "model_name":model_name,"confidence":float(confidence or 0)})
    _wj("forecasts.json", d[-1000:])

def read_forecasts(commodity_id=None, market_id=None) -> pd.DataFrame:
    d = _rj("forecasts.json", [])
    if not d: return pd.DataFrame()
    df = pd.DataFrame(d)
    df = df[df["forecast_date"]>=str(date.today())]
    if commodity_id: df=df[df["commodity_id"]==int(commodity_id)]
    if market_id:    df=df[df["market_id"]==int(market_id)]
    c = read_commodities(); m = read_markets()
    df = df.merge(c[["id","name"]].rename(columns={"id":"commodity_id","name":"commodity_name"}), on="commodity_id", how="left")
    df = df.merge(m[["id","market_name","state"]].rename(columns={"id":"market_id"}), on="market_id", how="left")
    return df.sort_values("forecast_date").reset_index(drop=True)

# ── AI Insights ───────────────────────────────────────────────────────────────
def save_insight(commodity_id, market_id, insight_date, insight_type, title, content, sentiment="neutral"):
    d = _rj("insights.json", [])
    d.append({"commodity_id":commodity_id,"market_id":market_id,
              "insight_date":str(insight_date),"insight_type":insight_type,
              "title":title,"content":content,"sentiment":sentiment})
    _wj("insights.json", d[-200:])

def read_insights(days=7) -> pd.DataFrame:
    d = _rj("insights.json", [])
    if not d: return pd.DataFrame()
    df = pd.DataFrame(d)
    df = df[df["insight_date"]>=str(date.today()-timedelta(days=days))]
    return df.sort_values("insight_date", ascending=False).reset_index(drop=True)

# ── Scheduler log ─────────────────────────────────────────────────────────────
def log_scheduler(job_name, status, records_processed=0, error_message=None):
    d = _rj("scheduler_log.json", [])
    d.append({"job_name":job_name,"status":status,"records_processed":records_processed,
              "error_message":error_message,"executed_at":str(date.today())})
    _wj("scheduler_log.json", d[-200:])

def read_scheduler_log(limit=50) -> pd.DataFrame:
    d = _rj("scheduler_log.json", [])
    if not d: return pd.DataFrame()
    return pd.DataFrame(d[-limit:][::-1])

# ── Analytics ─────────────────────────────────────────────────────────────────
def get_summary_stats() -> Dict[str, Any]:
    prices = _lp()
    comms  = read_commodities()
    mkts   = read_markets()
    anom   = _rj("anomalies.json", [])
    cutoff = str(date.today()-timedelta(days=30))
    return {
        "total_records":     len(prices),
        "total_commodities": len(comms),
        "total_markets":     len(mkts),
        "total_states":      mkts["state"].nunique() if not mkts.empty else 0,
        "latest_date":       prices["price_date"].max() if not prices.empty else "N/A",
        "anomalies_30d":     sum(1 for a in anom if a.get("detected_date","")>=cutoff),
    }

def get_state_analytics() -> pd.DataFrame:
    df = read_prices()
    if df.empty: return pd.DataFrame()
    return (df.groupby("state")
              .agg(num_markets=("market_name","nunique"),
                   num_commodities=("commodity_name","nunique"),
                   avg_price=("modal_price","mean"),
                   max_price=("modal_price","max"),
                   min_price=("modal_price","min"),
                   total_records=("modal_price","count"))
              .reset_index().sort_values("avg_price", ascending=False))

def get_top_commodities_by_volatility(limit=10) -> pd.DataFrame:
    df = read_prices()
    if df.empty: return pd.DataFrame()
    g = (df.groupby(["commodity_name","category"])["modal_price"]
           .agg(avg_price="mean", max_p="max", min_p="min", records="count").reset_index())
    g = g[g["records"]>5].copy()
    g["volatility_pct"] = (g["max_p"]-g["min_p"])/g["avg_price"].replace(0,1)*100
    return g.sort_values("volatility_pct", ascending=False).head(limit).reset_index(drop=True)

def get_price_trend(commodity_id: int, market_id: int, days: int = 90) -> pd.DataFrame:
    df = _lp()
    if df.empty: return pd.DataFrame()
    cutoff = str(date.today()-timedelta(days=days))
    df = df[(df["commodity_id"]==int(commodity_id))&
            (df["market_id"]==int(market_id))&
            (df["price_date"]>=cutoff)]
    return df[["price_date","min_price","max_price","modal_price","arrivals"]]\
             .sort_values("price_date").reset_index(drop=True)

def get_connection(): return None
