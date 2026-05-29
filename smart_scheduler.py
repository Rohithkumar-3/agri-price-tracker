"""
Smart Daily Scheduler with date-change detection.
- Checks if Agmarknet has uploaded NEW data (date changed)
- If today's data not yet uploaded, uses previous available date
- Retries at 7am, 8am, 9am, 10am until new data found
- Runs 24/7 — safe to deploy on any server/cloud
"""
import os, sys, time, logging
from datetime import date, timedelta, datetime
import requests

sys.path.insert(0, os.path.dirname(__file__))
from database import init_db, log_scheduler, get_connection
from data_pipeline import _get_api_key, AGMARKNET_BASE_URL, AGMARKNET_RESOURCE_ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scheduler.log", mode="a"),
    ]
)
log = logging.getLogger("smart_scheduler")


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_latest_api_date() -> str | None:
    """
    Ask Agmarknet API: what is the most recent date available?
    Returns date string 'YYYY-MM-DD' or None.
    """
    api_key = _get_api_key()
    if not api_key:
        return None
    url = f"{AGMARKNET_BASE_URL}/{AGMARKNET_RESOURCE_ID}"
    params = {"api-key": api_key, "format": "json", "limit": 1}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        records = resp.json().get("records", [])
        if not records:
            return None
        rec = records[0]
        for key in ["arrival_date", "Arrival_Date", "date", "Date"]:
            if key in rec:
                raw = rec[key]
                # Normalise dd/mm/yyyy → yyyy-mm-dd
                for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
                    try:
                        return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
                    except ValueError:
                        continue
    except Exception as e:
        log.warning(f"get_latest_api_date error: {e}")
    return None


def get_last_fetched_date() -> str | None:
    """What is the most recent date already stored in our DB?"""
    with get_connection() as conn:
        row = conn.execute("SELECT MAX(price_date) AS d FROM price_records WHERE source='Agmarknet'").fetchone()
        return row["d"] if row else None


def is_new_data_available() -> tuple[bool, str]:
    """
    Returns (is_new, best_date_to_fetch).
    is_new = True  → today's / newer date found on API
    is_new = False → API hasn't updated yet, use last known date
    """
    api_date      = get_latest_api_date()
    db_date       = get_last_fetched_date()
    today_str     = date.today().strftime("%Y-%m-%d")

    log.info(f"API latest date: {api_date} | DB latest date: {db_date} | Today: {today_str}")

    if api_date and api_date != db_date:
        return True, api_date          # new data on API
    if db_date:
        return False, db_date          # no new data, keep using last
    return False, today_str            # first run fallback


def run_fetch_job():
    """Main daily fetch with smart date detection."""
    from data_pipeline import fetch_and_store_prices
    from ml_models import run_anomaly_detection_all, run_forecast_all
    from ai_insights import bulk_generate_insights

    is_new, best_date = is_new_data_available()
    status_msg = "new data" if is_new else "no new API data (using last known)"
    log.info(f"Starting fetch job — {status_msg} — fetching for {best_date}")

    try:
        from datetime import datetime as dt
        target = dt.strptime(best_date, "%Y-%m-%d").date()
        count = fetch_and_store_prices(target_date=target, days_back=2)
        log.info(f"Fetch done: {count} records")
        log_scheduler("smart_fetch", "success", count,
                      f"date={best_date} new={is_new}")
    except Exception as e:
        log.error(f"Fetch failed: {e}")
        log_scheduler("smart_fetch", "error", 0, str(e))
        return

    # Anomaly detection
    try:
        n = run_anomaly_detection_all()
        log.info(f"Anomaly detection: {n} anomalies found")
        log_scheduler("anomaly_detection", "success", n)
    except Exception as e:
        log.error(f"Anomaly detection failed: {e}")
        log_scheduler("anomaly_detection", "error", 0, str(e))

    # ML Forecasts
    try:
        n = run_forecast_all(horizon=14)
        log.info(f"Forecasts: {n} records")
        log_scheduler("forecast", "success", n)
    except Exception as e:
        log.error(f"Forecast failed: {e}")
        log_scheduler("forecast", "error", 0, str(e))

    # AI Insights
    try:
        n = bulk_generate_insights(top_n=5)
        log.info(f"AI Insights: {n} generated")
        log_scheduler("ai_insights", "success", n)
    except Exception as e:
        log.error(f"AI insights failed: {e}")
        log_scheduler("ai_insights", "error", 0, str(e))


def run_with_retry():
    """
    Try fetching new data at 07:00, 08:00, 09:00, 10:00 IST.
    If new data found in any attempt, stop retrying for that day.
    """
    is_new, best_date = is_new_data_available()
    if is_new:
        log.info("New data confirmed. Running full pipeline now.")
        run_fetch_job()
        return True

    log.info("No new data yet from Agmarknet. Will retry in 1 hour.")
    return False


def start_scheduler():
    """24/7 loop — checks once per hour, runs full pipeline when new data arrives."""
    init_db()
    log.info("=" * 50)
    log.info("Smart Scheduler started — running 24/7")
    log.info("=" * 50)

    last_run_date = None  # track which calendar date we last successfully ran

    while True:
        now      = datetime.now()
        today    = date.today().isoformat()
        hour     = now.hour

        # Only actively try to fetch between 07:00–22:00 IST
        if 7 <= hour <= 22:
            if last_run_date != today:
                log.info(f"Attempting data fetch for {today} (hour={hour})")
                success = run_with_retry()
                if success:
                    last_run_date = today
                    log.info(f"Pipeline complete for {today}. Next run tomorrow.")
                else:
                    log.info("Will retry in 60 min.")
            # else: already ran today, sleep until tomorrow
        else:
            log.debug(f"Outside fetch window (hour={hour}). Sleeping.")

        time.sleep(3600)  # check every hour


if __name__ == "__main__":
    start_scheduler()
