"""
database.py — SQLite layer, Python 3.10-3.14 + Streamlit Cloud compatible.

Rules that make this work everywhere:
  1. Plain sqlite3.connect() — no autocommit/isolation_level flags
     (pandas pd.read_sql_query breaks with isolation_level=None on Py3.14)
  2. Every write function calls c.commit() explicitly before c.close()
  3. bulk_insert_prices uses explicit BEGIN/COMMIT for speed
  4. NO executescript() — replaced with individual execute() calls
  5. DB stored in /tmp on Streamlit Cloud (read-only source mount)
"""
import sqlite3, sys, os
import pandas as pd
from datetime import date
from typing import Optional, Dict, Any

# ── DB path ───────────────────────────────────────────────────────────────────
def _get_db_path() -> str:
    src = os.path.dirname(os.path.abspath(__file__))
    if not os.access(src, os.W_OK):
        os.makedirs("/tmp/agri_data", exist_ok=True)
        return "/tmp/agri_data/commodity_market.db"
    d = os.path.join(src, "data")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "commodity_market.db")

DB_PATH = _get_db_path()

# ── Connection ────────────────────────────────────────────────────────────────
def _conn():
    """
    Plain default connection — works with pandas AND with explicit commits.
    Do NOT change isolation_level or autocommit here; it breaks pandas on 3.14.
    """
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

# ── Schema — one execute() per statement, no executescript() ─────────────────
_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS commodities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        category TEXT NOT NULL,
        unit TEXT NOT NULL DEFAULT 'Quintal',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS markets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market_name TEXT NOT NULL,
        state TEXT NOT NULL,
        district TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(market_name, state))""",
    """CREATE TABLE IF NOT EXISTS price_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        commodity_id INTEGER NOT NULL,
        market_id INTEGER NOT NULL,
        price_date DATE NOT NULL,
        min_price REAL NOT NULL,
        max_price REAL NOT NULL,
        modal_price REAL NOT NULL,
        arrivals REAL DEFAULT 0,
        source TEXT DEFAULT 'Synthetic',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (commodity_id) REFERENCES commodities(id),
        FOREIGN KEY (market_id) REFERENCES markets(id),
        UNIQUE(commodity_id, market_id, price_date))""",
    """CREATE TABLE IF NOT EXISTS anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        price_record_id INTEGER,
        commodity_id INTEGER NOT NULL,
        market_id INTEGER NOT NULL,
        detected_date DATE NOT NULL,
        anomaly_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        deviation_pct REAL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        commodity_id INTEGER NOT NULL,
        market_id INTEGER NOT NULL,
        forecast_date DATE NOT NULL,
        predicted_price REAL NOT NULL,
        lower_bound REAL,
        upper_bound REAL,
        model_name TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(commodity_id, market_id, forecast_date))""",
    """CREATE TABLE IF NOT EXISTS ai_insights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        commodity_id INTEGER,
        market_id INTEGER,
        insight_date DATE NOT NULL,
        insight_type TEXT NOT NULL,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        sentiment TEXT DEFAULT 'neutral',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS scheduler_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_name TEXT NOT NULL,
        status TEXT NOT NULL,
        records_processed INTEGER DEFAULT 0,
        error_message TEXT,
        executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS app_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
    "CREATE INDEX IF NOT EXISTS idx_price_date      ON price_records(price_date)",
    "CREATE INDEX IF NOT EXISTS idx_price_commodity ON price_records(commodity_id)",
    "CREATE INDEX IF NOT EXISTS idx_price_market    ON price_records(market_id)",
]

def init_db():
    c = _conn()
    try:
        for stmt in _SCHEMA:
            c.execute(stmt)
        c.commit()
    finally:
        c.close()

# ── App meta ──────────────────────────────────────────────────────────────────
def get_meta(key: str) -> Optional[str]:
    c = _conn()
    try:
        row = c.execute("SELECT value FROM app_meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None
    except Exception:
        return None
    finally:
        c.close()

def set_meta(key: str, value: str):
    c = _conn()
    try:
        c.execute(
            "INSERT INTO app_meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP",
            (key, value))
        c.commit()
    finally:
        c.close()

# ── Commodities ───────────────────────────────────────────────────────────────
def create_commodity(name: str, category: str, unit: str = "Quintal") -> int:
    c = _conn()
    try:
        c.execute("INSERT OR IGNORE INTO commodities(name,category,unit) VALUES(?,?,?)",
                  (name, category, unit))
        c.commit()
        return c.execute("SELECT id FROM commodities WHERE name=?", (name,)).fetchone()["id"]
    finally:
        c.close()

def read_commodities() -> pd.DataFrame:
    c = _conn()
    try:
        return pd.read_sql_query("SELECT * FROM commodities ORDER BY name", c)
    finally:
        c.close()

def update_commodity(cid: int, name=None, category=None, unit=None):
    fields, vals = [], []
    if name:     fields.append("name=?");     vals.append(name)
    if category: fields.append("category=?"); vals.append(category)
    if unit:     fields.append("unit=?");     vals.append(unit)
    if not fields: return
    c = _conn()
    try:
        c.execute(f"UPDATE commodities SET {','.join(fields)} WHERE id=?", vals + [cid])
        c.commit()
    finally:
        c.close()

def delete_commodity(cid: int):
    c = _conn()
    try:
        c.execute("DELETE FROM commodities WHERE id=?", (cid,))
        c.commit()
    finally:
        c.close()

# ── Markets ───────────────────────────────────────────────────────────────────
def create_market(market_name: str, state: str, district: str) -> int:
    c = _conn()
    try:
        c.execute("INSERT OR IGNORE INTO markets(market_name,state,district) VALUES(?,?,?)",
                  (market_name, state, district))
        c.commit()
        return c.execute(
            "SELECT id FROM markets WHERE market_name=? AND state=?",
            (market_name, state)).fetchone()["id"]
    finally:
        c.close()

def read_markets() -> pd.DataFrame:
    c = _conn()
    try:
        return pd.read_sql_query("SELECT * FROM markets ORDER BY state, market_name", c)
    finally:
        c.close()

def delete_market(mid: int):
    c = _conn()
    try:
        c.execute("DELETE FROM markets WHERE id=?", (mid,))
        c.commit()
    finally:
        c.close()

# ── Price records ─────────────────────────────────────────────────────────────
def upsert_price_record(commodity_id, market_id, price_date,
                        min_price, max_price, modal_price,
                        arrivals=0, source="Synthetic"):
    c = _conn()
    try:
        c.execute("""
            INSERT INTO price_records
                (commodity_id,market_id,price_date,min_price,max_price,modal_price,arrivals,source)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(commodity_id,market_id,price_date) DO UPDATE SET
                min_price=excluded.min_price, max_price=excluded.max_price,
                modal_price=excluded.modal_price, arrivals=excluded.arrivals,
                source=excluded.source
        """, (commodity_id, market_id, price_date,
              float(min_price), float(max_price), float(modal_price),
              float(arrivals), source))
        c.commit()
    finally:
        c.close()

def bulk_insert_prices(records: list):
    """Fast bulk insert. records = list of 8-tuples."""
    if not records:
        return
    clean = [(int(r[0]), int(r[1]), str(r[2]),
              float(r[3]), float(r[4]), float(r[5]),
              float(r[6]), str(r[7])) for r in records]
    c = _conn()
    try:
        c.execute("BEGIN")
        c.executemany("""
            INSERT OR IGNORE INTO price_records
                (commodity_id,market_id,price_date,min_price,max_price,modal_price,arrivals,source)
            VALUES (?,?,?,?,?,?,?,?)
        """, clean)
        c.execute("COMMIT")
    except Exception as e:
        try: c.execute("ROLLBACK")
        except Exception: pass
        raise
    finally:
        c.close()

def read_prices(commodity_id=None, market_id=None,
                start_date=None, end_date=None) -> pd.DataFrame:
    q = """SELECT pr.*, c.name AS commodity_name, c.category, c.unit,
                  m.market_name, m.state, m.district
           FROM price_records pr
           JOIN commodities c ON pr.commodity_id=c.id
           JOIN markets m ON pr.market_id=m.id WHERE 1=1"""
    p = []
    if commodity_id: q += " AND pr.commodity_id=?"; p.append(commodity_id)
    if market_id:    q += " AND pr.market_id=?";    p.append(market_id)
    if start_date:   q += " AND pr.price_date>=?";  p.append(start_date)
    if end_date:     q += " AND pr.price_date<=?";  p.append(end_date)
    q += " ORDER BY pr.price_date DESC"
    c = _conn()
    try:
        return pd.read_sql_query(q, c, params=p)
    finally:
        c.close()

def delete_price_record(record_id: int):
    c = _conn()
    try:
        c.execute("DELETE FROM price_records WHERE id=?", (record_id,))
        c.commit()
    finally:
        c.close()

# ── Anomalies ─────────────────────────────────────────────────────────────────
def save_anomaly(commodity_id, market_id, detected_date, anomaly_type,
                 severity, deviation_pct, description, price_record_id=None):
    c = _conn()
    try:
        c.execute("""INSERT INTO anomalies
            (price_record_id,commodity_id,market_id,detected_date,
             anomaly_type,severity,deviation_pct,description)
            VALUES (?,?,?,?,?,?,?,?)""",
            (price_record_id, commodity_id, market_id, detected_date,
             anomaly_type, severity, deviation_pct, description))
        c.commit()
    finally:
        c.close()

def read_anomalies(days: int = 30) -> pd.DataFrame:
    c = _conn()
    try:
        return pd.read_sql_query("""
            SELECT a.*, c.name AS commodity_name, m.market_name, m.state
            FROM anomalies a
            JOIN commodities c ON a.commodity_id=c.id
            JOIN markets m ON a.market_id=m.id
            WHERE a.detected_date >= date('now',?)
            ORDER BY a.detected_date DESC
        """, c, params=[f"-{days} days"])
    finally:
        c.close()

# ── Forecasts ─────────────────────────────────────────────────────────────────
def save_forecast(commodity_id, market_id, forecast_date, predicted_price,
                  lower_bound=None, upper_bound=None,
                  model_name="RandomForest", confidence=None):
    c = _conn()
    try:
        c.execute("""INSERT INTO forecasts
            (commodity_id,market_id,forecast_date,predicted_price,
             lower_bound,upper_bound,model_name,confidence)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(commodity_id,market_id,forecast_date) DO UPDATE SET
                predicted_price=excluded.predicted_price,
                lower_bound=excluded.lower_bound,
                upper_bound=excluded.upper_bound""",
            (commodity_id, market_id, forecast_date, predicted_price,
             lower_bound, upper_bound, model_name, confidence))
        c.commit()
    finally:
        c.close()

def read_forecasts(commodity_id=None, market_id=None) -> pd.DataFrame:
    q = """SELECT f.*, c.name AS commodity_name, m.market_name, m.state
           FROM forecasts f
           JOIN commodities c ON f.commodity_id=c.id
           JOIN markets m ON f.market_id=m.id
           WHERE f.forecast_date >= date('now')"""
    p = []
    if commodity_id: q += " AND f.commodity_id=?"; p.append(commodity_id)
    if market_id:    q += " AND f.market_id=?";    p.append(market_id)
    q += " ORDER BY f.forecast_date"
    c = _conn()
    try:
        return pd.read_sql_query(q, c, params=p)
    finally:
        c.close()

# ── AI Insights ───────────────────────────────────────────────────────────────
def save_insight(commodity_id, market_id, insight_date,
                 insight_type, title, content, sentiment="neutral"):
    c = _conn()
    try:
        c.execute("""INSERT INTO ai_insights
            (commodity_id,market_id,insight_date,insight_type,title,content,sentiment)
            VALUES (?,?,?,?,?,?,?)""",
            (commodity_id, market_id, insight_date,
             insight_type, title, content, sentiment))
        c.commit()
    finally:
        c.close()

def read_insights(days: int = 7) -> pd.DataFrame:
    c = _conn()
    try:
        return pd.read_sql_query("""
            SELECT ai.*, c.name AS commodity_name, m.market_name
            FROM ai_insights ai
            LEFT JOIN commodities c ON ai.commodity_id=c.id
            LEFT JOIN markets m ON ai.market_id=m.id
            WHERE ai.insight_date >= date('now',?)
            ORDER BY ai.created_at DESC
        """, c, params=[f"-{days} days"])
    finally:
        c.close()

# ── Scheduler log ─────────────────────────────────────────────────────────────
def log_scheduler(job_name, status, records_processed=0, error_message=None):
    c = _conn()
    try:
        c.execute("""INSERT INTO scheduler_log
            (job_name,status,records_processed,error_message) VALUES (?,?,?,?)""",
            (job_name, status, records_processed, error_message))
        c.commit()
    finally:
        c.close()

def read_scheduler_log(limit: int = 50) -> pd.DataFrame:
    c = _conn()
    try:
        return pd.read_sql_query(
            f"SELECT * FROM scheduler_log ORDER BY executed_at DESC LIMIT {limit}", c)
    finally:
        c.close()

# ── Analytics ─────────────────────────────────────────────────────────────────
def get_summary_stats() -> Dict[str, Any]:
    c = _conn()
    try:
        return {
            "total_records":     c.execute("SELECT COUNT(*) FROM price_records").fetchone()[0],
            "total_commodities": c.execute("SELECT COUNT(*) FROM commodities").fetchone()[0],
            "total_markets":     c.execute("SELECT COUNT(*) FROM markets").fetchone()[0],
            "total_states":      c.execute("SELECT COUNT(DISTINCT state) FROM markets").fetchone()[0],
            "latest_date":       c.execute("SELECT MAX(price_date) FROM price_records").fetchone()[0],
            "anomalies_30d":     c.execute(
                "SELECT COUNT(*) FROM anomalies WHERE detected_date >= date('now','-30 days')"
            ).fetchone()[0],
        }
    finally:
        c.close()

def get_state_analytics() -> pd.DataFrame:
    c = _conn()
    try:
        return pd.read_sql_query("""
            SELECT m.state,
                   COUNT(DISTINCT m.id) AS num_markets,
                   COUNT(DISTINCT pr.commodity_id) AS num_commodities,
                   AVG(pr.modal_price) AS avg_price,
                   MAX(pr.modal_price) AS max_price,
                   MIN(pr.modal_price) AS min_price,
                   COUNT(pr.id) AS total_records
            FROM markets m JOIN price_records pr ON m.id=pr.market_id
            GROUP BY m.state ORDER BY avg_price DESC
        """, c)
    finally:
        c.close()

def get_top_commodities_by_volatility(limit: int = 10) -> pd.DataFrame:
    c = _conn()
    try:
        return pd.read_sql_query("""
            SELECT c.name AS commodity_name, c.category,
                   AVG(pr.modal_price) AS avg_price,
                   (MAX(pr.modal_price)-MIN(pr.modal_price))
                       /NULLIF(AVG(pr.modal_price),0)*100 AS volatility_pct,
                   COUNT(pr.id) AS records
            FROM commodities c JOIN price_records pr ON c.id=pr.commodity_id
            GROUP BY c.id HAVING records>5
            ORDER BY volatility_pct DESC LIMIT ?
        """, c, params=[limit])
    finally:
        c.close()

def get_price_trend(commodity_id: int, market_id: int, days: int = 90) -> pd.DataFrame:
    c = _conn()
    try:
        return pd.read_sql_query("""
            SELECT price_date,min_price,max_price,modal_price,arrivals
            FROM price_records
            WHERE commodity_id=? AND market_id=?
              AND price_date >= date('now',?)
            ORDER BY price_date
        """, c, params=[commodity_id, market_id, f"-{days} days"])
    finally:
        c.close()

def get_connection():
    return _conn()
