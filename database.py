"""
Database layer - SQLite with full CRUD operations.
Streamlit Cloud compatible — uses /tmp for writable storage.
"""
import sqlite3
import pandas as pd
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import os

# ── Writable path that works on Streamlit Cloud AND locally ──────────────────
# Streamlit Cloud: /mount/src/ is read-only, /tmp is writable
# Local: use project data/ folder
def _get_db_path():
    # If running on Streamlit Cloud (read-only source mount)
    if os.path.exists("/tmp") and not os.access(os.path.dirname(__file__), os.W_OK):
        os.makedirs("/tmp/agri_data", exist_ok=True)
        return "/tmp/agri_data/commodity_market.db"
    # Local development — use data/ folder next to this file
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "commodity_market.db")

DB_PATH = _get_db_path()


def get_connection():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES,
                           check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL mode fails on some cloud filesystems — use DELETE mode (safe everywhere)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    """Initialize all tables."""
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS commodities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            unit TEXT NOT NULL DEFAULT 'Quintal',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS markets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_name TEXT NOT NULL,
            state TEXT NOT NULL,
            district TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(market_name, state)
        );

        CREATE TABLE IF NOT EXISTS price_records (
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
            UNIQUE(commodity_id, market_id, price_date)
        );

        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            price_record_id INTEGER,
            commodity_id INTEGER NOT NULL,
            market_id INTEGER NOT NULL,
            detected_date DATE NOT NULL,
            anomaly_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            deviation_pct REAL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS forecasts (
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
            UNIQUE(commodity_id, market_id, forecast_date)
        );

        CREATE TABLE IF NOT EXISTS ai_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            commodity_id INTEGER,
            market_id INTEGER,
            insight_date DATE NOT NULL,
            insight_type TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            sentiment TEXT DEFAULT 'neutral',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS scheduler_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_name TEXT NOT NULL,
            status TEXT NOT NULL,
            records_processed INTEGER DEFAULT 0,
            error_message TEXT,
            executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS app_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_price_date       ON price_records(price_date);
        CREATE INDEX IF NOT EXISTS idx_price_commodity  ON price_records(commodity_id);
        CREATE INDEX IF NOT EXISTS idx_price_market     ON price_records(market_id);
        """)
        conn.commit()


# ── App meta (used to track seeding state) ────────────────────────────────────

def get_meta(key: str) -> Optional[str]:
    try:
        with get_connection() as conn:
            row = conn.execute("SELECT value FROM app_meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else None
    except Exception:
        return None


def set_meta(key: str, value: str):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO app_meta (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value,
            updated_at=CURRENT_TIMESTAMP
        """, (key, value))
        conn.commit()


# ── Commodity CRUD ────────────────────────────────────────────────────────────

def create_commodity(name: str, category: str, unit: str = "Quintal") -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO commodities (name, category, unit) VALUES (?,?,?)",
            (name, category, unit)
        )
        conn.commit()
        if cur.lastrowid:
            return cur.lastrowid
        row = conn.execute("SELECT id FROM commodities WHERE name=?", (name,)).fetchone()
        return row["id"]


def read_commodities() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM commodities ORDER BY name", conn)


def update_commodity(cid: int, name: str = None, category: str = None, unit: str = None):
    fields, vals = [], []
    if name:     fields.append("name=?");     vals.append(name)
    if category: fields.append("category=?"); vals.append(category)
    if unit:     fields.append("unit=?");     vals.append(unit)
    if not fields:
        return
    vals.append(cid)
    with get_connection() as conn:
        conn.execute(f"UPDATE commodities SET {', '.join(fields)} WHERE id=?", vals)
        conn.commit()


def delete_commodity(cid: int):
    with get_connection() as conn:
        conn.execute("DELETE FROM commodities WHERE id=?", (cid,))
        conn.commit()


# ── Market CRUD ───────────────────────────────────────────────────────────────

def create_market(market_name: str, state: str, district: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO markets (market_name, state, district) VALUES (?,?,?)",
            (market_name, state, district)
        )
        conn.commit()
        if cur.lastrowid:
            return cur.lastrowid
        row = conn.execute(
            "SELECT id FROM markets WHERE market_name=? AND state=?",
            (market_name, state)
        ).fetchone()
        return row["id"]


def read_markets() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM markets ORDER BY state, market_name", conn)


def delete_market(mid: int):
    with get_connection() as conn:
        conn.execute("DELETE FROM markets WHERE id=?", (mid,))
        conn.commit()


# ── Price Records CRUD ────────────────────────────────────────────────────────

def upsert_price_record(commodity_id, market_id, price_date,
                        min_price, max_price, modal_price,
                        arrivals=0, source="Synthetic"):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO price_records
                (commodity_id, market_id, price_date,
                 min_price, max_price, modal_price, arrivals, source)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(commodity_id, market_id, price_date)
            DO UPDATE SET
                min_price    = excluded.min_price,
                max_price    = excluded.max_price,
                modal_price  = excluded.modal_price,
                arrivals     = excluded.arrivals,
                source       = excluded.source
        """, (commodity_id, market_id, price_date,
              float(min_price), float(max_price), float(modal_price),
              float(arrivals), source))
        conn.commit()


def read_prices(commodity_id=None, market_id=None,
                start_date=None, end_date=None) -> pd.DataFrame:
    query = """
        SELECT pr.*, c.name AS commodity_name, c.category, c.unit,
               m.market_name, m.state, m.district
        FROM price_records pr
        JOIN commodities c ON pr.commodity_id = c.id
        JOIN markets m     ON pr.market_id    = m.id
        WHERE 1=1
    """
    params = []
    if commodity_id: query += " AND pr.commodity_id=?"; params.append(commodity_id)
    if market_id:    query += " AND pr.market_id=?";    params.append(market_id)
    if start_date:   query += " AND pr.price_date>=?";  params.append(start_date)
    if end_date:     query += " AND pr.price_date<=?";  params.append(end_date)
    query += " ORDER BY pr.price_date DESC"
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


def delete_price_record(record_id: int):
    with get_connection() as conn:
        conn.execute("DELETE FROM price_records WHERE id=?", (record_id,))
        conn.commit()


# ── Anomaly CRUD ──────────────────────────────────────────────────────────────

def save_anomaly(commodity_id, market_id, detected_date, anomaly_type,
                 severity, deviation_pct, description, price_record_id=None):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO anomalies
                (price_record_id, commodity_id, market_id, detected_date,
                 anomaly_type, severity, deviation_pct, description)
            VALUES (?,?,?,?,?,?,?,?)
        """, (price_record_id, commodity_id, market_id, detected_date,
              anomaly_type, severity, deviation_pct, description))
        conn.commit()


def read_anomalies(days: int = 30) -> pd.DataFrame:
    query = """
        SELECT a.*, c.name AS commodity_name, m.market_name, m.state
        FROM anomalies a
        JOIN commodities c ON a.commodity_id = c.id
        JOIN markets m     ON a.market_id    = m.id
        WHERE a.detected_date >= date('now', ?)
        ORDER BY a.detected_date DESC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=[f"-{days} days"])


# ── Forecast CRUD ─────────────────────────────────────────────────────────────

def save_forecast(commodity_id, market_id, forecast_date, predicted_price,
                  lower_bound=None, upper_bound=None,
                  model_name="RandomForest", confidence=None):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO forecasts
                (commodity_id, market_id, forecast_date, predicted_price,
                 lower_bound, upper_bound, model_name, confidence)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(commodity_id, market_id, forecast_date)
            DO UPDATE SET
                predicted_price = excluded.predicted_price,
                lower_bound     = excluded.lower_bound,
                upper_bound     = excluded.upper_bound
        """, (commodity_id, market_id, forecast_date, predicted_price,
              lower_bound, upper_bound, model_name, confidence))
        conn.commit()


def read_forecasts(commodity_id=None, market_id=None) -> pd.DataFrame:
    query = """
        SELECT f.*, c.name AS commodity_name, m.market_name, m.state
        FROM forecasts f
        JOIN commodities c ON f.commodity_id = c.id
        JOIN markets m     ON f.market_id    = m.id
        WHERE f.forecast_date >= date('now')
    """
    params = []
    if commodity_id: query += " AND f.commodity_id=?"; params.append(commodity_id)
    if market_id:    query += " AND f.market_id=?";    params.append(market_id)
    query += " ORDER BY f.forecast_date"
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


# ── AI Insights CRUD ──────────────────────────────────────────────────────────

def save_insight(commodity_id, market_id, insight_date,
                 insight_type, title, content, sentiment="neutral"):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO ai_insights
                (commodity_id, market_id, insight_date,
                 insight_type, title, content, sentiment)
            VALUES (?,?,?,?,?,?,?)
        """, (commodity_id, market_id, insight_date,
              insight_type, title, content, sentiment))
        conn.commit()


def read_insights(days: int = 7) -> pd.DataFrame:
    query = """
        SELECT ai.*, c.name AS commodity_name, m.market_name
        FROM ai_insights ai
        LEFT JOIN commodities c ON ai.commodity_id = c.id
        LEFT JOIN markets m     ON ai.market_id    = m.id
        WHERE ai.insight_date >= date('now', ?)
        ORDER BY ai.created_at DESC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=[f"-{days} days"])


# ── Scheduler Log ─────────────────────────────────────────────────────────────

def log_scheduler(job_name, status, records_processed=0, error_message=None):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO scheduler_log
                (job_name, status, records_processed, error_message)
            VALUES (?,?,?,?)
        """, (job_name, status, records_processed, error_message))
        conn.commit()


def read_scheduler_log(limit: int = 50) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(
            f"SELECT * FROM scheduler_log ORDER BY executed_at DESC LIMIT {limit}", conn
        )


# ── Analytics ─────────────────────────────────────────────────────────────────

def get_summary_stats() -> Dict[str, Any]:
    with get_connection() as conn:
        return {
            "total_records":    conn.execute("SELECT COUNT(*) FROM price_records").fetchone()[0],
            "total_commodities":conn.execute("SELECT COUNT(*) FROM commodities").fetchone()[0],
            "total_markets":    conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0],
            "total_states":     conn.execute("SELECT COUNT(DISTINCT state) FROM markets").fetchone()[0],
            "latest_date":      conn.execute("SELECT MAX(price_date) FROM price_records").fetchone()[0],
            "anomalies_30d":    conn.execute(
                "SELECT COUNT(*) FROM anomalies WHERE detected_date >= date('now','-30 days')"
            ).fetchone()[0],
        }


def get_state_analytics() -> pd.DataFrame:
    query = """
        SELECT m.state,
               COUNT(DISTINCT m.id)           AS num_markets,
               COUNT(DISTINCT pr.commodity_id) AS num_commodities,
               AVG(pr.modal_price)             AS avg_price,
               MAX(pr.modal_price)             AS max_price,
               MIN(pr.modal_price)             AS min_price,
               COUNT(pr.id)                   AS total_records
        FROM markets m
        JOIN price_records pr ON m.id = pr.market_id
        GROUP BY m.state
        ORDER BY avg_price DESC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)


def get_top_commodities_by_volatility(limit: int = 10) -> pd.DataFrame:
    query = """
        SELECT c.name AS commodity_name, c.category,
               AVG(pr.modal_price) AS avg_price,
               (MAX(pr.modal_price) - MIN(pr.modal_price))
                   / NULLIF(AVG(pr.modal_price), 0) * 100 AS volatility_pct,
               COUNT(pr.id) AS records
        FROM commodities c
        JOIN price_records pr ON c.id = pr.commodity_id
        GROUP BY c.id
        HAVING records > 5
        ORDER BY volatility_pct DESC
        LIMIT ?
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=[limit])


def get_price_trend(commodity_id: int, market_id: int, days: int = 90) -> pd.DataFrame:
    query = """
        SELECT price_date, min_price, max_price, modal_price, arrivals
        FROM price_records
        WHERE commodity_id=? AND market_id=?
          AND price_date >= date('now', ?)
        ORDER BY price_date
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn,
                                 params=[commodity_id, market_id, f"-{days} days"])
