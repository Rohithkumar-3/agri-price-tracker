"""
AI Insights - uses FREE APIs (Gemini or Groq).
No Anthropic/paid API needed.

Priority order:
  1. Google Gemini (free, 500 req/day) — aistudio.google.com
  2. Groq (free, 14400 req/day)        — console.groq.com
  3. Rule-based fallback               — always works, no key needed
"""
import os, sys, json, requests
from datetime import date, timedelta
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from database import read_prices, read_anomalies, save_insight, read_commodities, read_markets


# ── Key loaders ───────────────────────────────────────────────────────────────

def _load_key(name: str) -> str:
    """Load key from env var or .env file."""
    val = os.environ.get(name, "").strip()
    if val and val != "your_key_here":
        return val
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{name}="):
                    v = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if v and v != "your_key_here":
                        return v
    return ""


# ── Provider: Google Gemini (FREE — no credit card) ───────────────────────────

def _call_gemini(prompt: str) -> str:
    api_key = _load_key("GEMINI_API_KEY")
    if not api_key:
        return ""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 300, "temperature": 0.4},
        "systemInstruction": {
            "parts": [{"text": (
                "You are an agricultural market analyst for Indian commodity markets. "
                "Give clear, practical advice in plain English. "
                "Keep it under 120 words. No bullet points or headers."
            )}]
        }
    }
    try:
        r = requests.post(url, json=body, timeout=15)
        if r.status_code == 200:
            candidates = r.json().get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                return " ".join(p.get("text", "") for p in parts).strip()
        elif r.status_code == 429:
            print("[Gemini] Rate limit hit — trying Groq next")
    except Exception as e:
        print(f"[Gemini] Error: {e}")
    return ""


# ── Provider: Groq (FREE — no credit card) ────────────────────────────────────

def _call_groq(prompt: str) -> str:
    api_key = _load_key("GROQ_API_KEY")
    if not api_key:
        return ""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an agricultural market analyst for Indian commodity markets. "
                    "Give clear, practical advice in plain English. "
                    "Keep it under 120 words. No bullet points or headers."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.4,
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=15)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        elif r.status_code == 429:
            print("[Groq] Rate limit hit — using rule-based fallback")
    except Exception as e:
        print(f"[Groq] Error: {e}")
    return ""


# ── Rule-based fallback (no key needed, always works) ────────────────────────

def _rule_based_insight(commodity: str, modal: float, avg: float,
                        week_chg: float, month_chg: float,
                        min_p: float, max_p: float, market: str) -> str:
    spread_pct = (max_p - min_p) / modal * 100 if modal else 0
    vs_avg     = (modal - avg) / avg * 100 if avg else 0

    # Trend direction
    if week_chg > 8:
        trend = f"Prices have risen sharply by {week_chg:.1f}% this week."
        why   = "This likely reflects reduced market arrivals, increased demand from traders, or supply disruption."
        advice = "If you have stock, this may be a good time to sell at the current high."
    elif week_chg > 3:
        trend = f"Prices are moving up by {week_chg:.1f}% this week."
        why   = "Moderate buying interest in the market. Arrivals may be slightly lower than usual."
        advice = "Monitor for 2–3 more days. Prices could rise further before correcting."
    elif week_chg < -8:
        trend = f"Prices have dropped sharply by {abs(week_chg):.1f}% this week."
        why   = "Heavy arrivals at the mandi or reduced buyer demand is pushing prices down."
        advice = "If possible, hold back stock for a few days and watch if prices recover."
    elif week_chg < -3:
        trend = f"Prices are easing down by {abs(week_chg):.1f}% this week."
        why   = "Supply is slightly above demand at current levels."
        advice = "No immediate action needed. Watch the next 3–5 days for direction."
    else:
        trend = f"Prices are stable this week (change: {week_chg:+.1f}%)."
        why   = "Supply and demand are balanced at current levels."
        advice = "No urgent action required. Sell when convenient."

    # vs 30-day average
    if vs_avg > 10:
        avg_note = f"At ₹{modal:,.0f}, the price is {vs_avg:.0f}% above the 30-day average — above normal range."
    elif vs_avg < -10:
        avg_note = f"At ₹{modal:,.0f}, the price is {abs(vs_avg):.0f}% below the 30-day average — below normal range."
    else:
        avg_note = f"At ₹{modal:,.0f}, the price is within the normal 30-day range (avg ₹{avg:,.0f})."

    # Spread note
    if spread_pct > 15:
        spread_note = f"Today's price spread is wide (₹{min_p:,.0f}–₹{max_p:,.0f}), showing price uncertainty."
    else:
        spread_note = ""

    parts = [trend, why, avg_note, advice]
    if spread_note:
        parts.append(spread_note)
    return " ".join(parts)


# ── Main dispatcher ───────────────────────────────────────────────────────────

def _call_ai(prompt: str, commodity: str = "", modal: float = 0,
             avg: float = 0, week_chg: float = 0, month_chg: float = 0,
             min_p: float = 0, max_p: float = 0, market: str = "") -> str:
    """Try Gemini → Groq → rule-based, return first successful result."""

    # 1. Try Gemini
    result = _call_gemini(prompt)
    if result:
        return result

    # 2. Try Groq
    result = _call_groq(prompt)
    if result:
        return result

    # 3. Rule-based fallback — always works
    return _rule_based_insight(commodity, modal, avg, week_chg, month_chg, min_p, max_p, market)


# ── Public functions ──────────────────────────────────────────────────────────

def generate_price_movement_insight(commodity_id: int, market_id: int) -> str:
    df = read_prices(commodity_id=commodity_id, market_id=market_id,
                     start_date=str(date.today() - timedelta(days=45)))
    if df.empty or len(df) < 7:
        return "Not enough data to generate insight. Try after a few more days of price data."

    df        = df.sort_values("price_date")
    latest    = df.iloc[-1]
    week_ago  = df.iloc[-7] if len(df) >= 7  else df.iloc[0]
    month_ago = df.iloc[-30] if len(df) >= 30 else df.iloc[0]
    modal     = latest["modal_price"]
    avg_30    = df.tail(30)["modal_price"].mean()
    week_chg  = (modal - week_ago["modal_price"])  / week_ago["modal_price"]  * 100
    month_chg = (modal - month_ago["modal_price"]) / month_ago["modal_price"] * 100
    min_p     = df.tail(7)["min_price"].min()
    max_p     = df.tail(7)["max_price"].max()

    prompt = (
        f"Commodity: {latest['commodity_name']} at {latest['market_name']}, {latest['state']}\n"
        f"Today's modal price: ₹{modal:.0f}/quintal\n"
        f"7-day change: {week_chg:+.1f}%\n"
        f"30-day change: {month_chg:+.1f}%\n"
        f"30-day average: ₹{avg_30:.0f}/quintal\n"
        f"7-day price range: ₹{min_p:.0f} to ₹{max_p:.0f}\n\n"
        f"In plain English, explain what is happening with this price and give practical "
        f"advice to farmers on whether to sell now or wait."
    )
    return _call_ai(
        prompt,
        commodity=latest["commodity_name"], modal=modal, avg=avg_30,
        week_chg=week_chg, month_chg=month_chg,
        min_p=min_p, max_p=max_p, market=latest["market_name"]
    )


def generate_market_summary_insight(state: str = None) -> str:
    from database import get_top_commodities_by_volatility
    vol_df   = get_top_commodities_by_volatility(5)
    vol_list = ", ".join(
        f"{r['commodity_name']} ({r['volatility_pct']:.0f}% swing)"
        for _, r in vol_df.iterrows()
    ) if not vol_df.empty else "data unavailable"

    prompt = (
        f"Indian commodity market{'in ' + state if state else ''} overview:\n"
        f"Most volatile commodities: {vol_list}\n\n"
        f"Write a 2–3 sentence market summary with practical advice for farmers."
    )

    result = _call_gemini(prompt) or _call_groq(prompt)
    if result:
        return result

    # Fallback summary
    if not vol_df.empty:
        top = vol_df.iloc[0]["commodity_name"]
        return (
            f"Markets are active with {top} showing the highest price swings this season. "
            f"Farmers should monitor mandi arrival data and government procurement announcements. "
            f"Selling in small lots rather than all at once can help manage price risk."
        )
    return "Market data is being updated. Check back shortly for the latest summary."


def generate_anomaly_insight(anomaly: dict) -> str:
    dev    = anomaly.get("deviation_pct", 0)
    cname  = anomaly.get("commodity_name", "commodity")
    market = anomaly.get("market_name", "market")
    sev    = anomaly.get("severity", "")

    prompt = (
        f"Unusual price movement detected:\n"
        f"Commodity: {cname} at {market}, {anomaly.get('state', '')}\n"
        f"Type: {anomaly.get('anomaly_type')}, Severity: {sev}\n"
        f"Price deviation: {dev:+.1f}%\n"
        f"Details: {anomaly.get('description', '')}\n\n"
        f"In 2–3 plain English sentences, explain what likely caused this and what farmers should do."
    )

    result = _call_gemini(prompt) or _call_groq(prompt)
    if result:
        return result

    # Rule-based anomaly insight
    if dev > 15:
        return (
            f"{cname} prices spiked {dev:.0f}% above normal at {market}. "
            f"This could be due to low arrivals, high trader demand, or supply chain disruption. "
            f"Farmers with ready stock may benefit from selling now while prices are high."
        )
    elif dev < -15:
        return (
            f"{cname} prices dropped {abs(dev):.0f}% below normal at {market}. "
            f"Heavy arrivals or reduced demand is likely pushing prices down. "
            f"If storage is available, holding back stock for a few days may help recover value."
        )
    return (
        f"An unusual price pattern was detected for {cname} at {market}. "
        f"Monitor prices closely over the next 2–3 days before making selling decisions."
    )


def bulk_generate_insights(top_n: int = 5):
    from database import get_connection
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT commodity_id, market_id, COUNT(*) as cnt
            FROM price_records
            GROUP BY commodity_id, market_id
            ORDER BY cnt DESC LIMIT ?
        """, (top_n,)).fetchall()

    count = 0
    for row in rows:
        cid, mid = row["commodity_id"], row["market_id"]
        text = generate_price_movement_insight(cid, mid)
        df   = read_prices(commodity_id=cid, market_id=mid)
        if not df.empty:
            last      = df.iloc[0]
            sentiment = ("positive" if any(w in text.lower() for w in ["good time to sell","risen","high"])
                         else "negative" if any(w in text.lower() for w in ["drop","fell","low","hold"])
                         else "neutral")
            save_insight(cid, mid, str(date.today()), "price_movement",
                         f"{last['commodity_name']} @ {last['market_name']} — Analysis",
                         text, sentiment)
            count += 1

    summary = generate_market_summary_insight()
    save_insight(None, None, str(date.today()), "market_summary",
                 "Daily Market Overview", summary, "neutral")
    return count + 1
