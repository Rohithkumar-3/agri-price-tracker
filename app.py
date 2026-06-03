"""
Agri Price Tracker — Commodity Market Dashboard
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from database import (
    init_db, get_summary_stats, read_commodities, read_markets,
    read_prices, read_anomalies, read_forecasts, read_insights,
    get_state_analytics, get_top_commodities_by_volatility,
    get_price_trend, read_scheduler_log, get_meta, set_meta,
)
from data_pipeline import fetch_and_store_prices, seed_reference_data
from ml_models import forecast_prices, detect_anomalies
from ai_insights import generate_price_movement_insight

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agri Price Tracker 🌾",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Add right after st.set_page_config
import os
from pathlib import Path

# Serve Google verification file
verify_files = list(Path(".").glob("google4a2feef810cce88b.html"))
if verify_files:
    with open(verify_files[0]) as f:
        content = f.read()
    st.markdown(content, unsafe_allow_html=True)
# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; }

section[data-testid="stSidebar"] { background: #161b27 !important; border-right: 1px solid #1e2d40; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div { background: #1e2d40 !important; border: 1px solid #2d4a6b !important; color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stButton > button { background: #16a34a !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }

.stTabs [data-baseweb="tab-list"] { background: #161b27; border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid #1e2d40; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #94a3b8; border-radius: 8px; font-weight: 500; padding: 8px 18px; border: none; }
.stTabs [aria-selected="true"] { background: #16a34a !important; color: white !important; font-weight: 600 !important; }

[data-testid="metric-container"] { background: #161b27; border: 1px solid #1e2d40; border-radius: 12px; padding: 16px; }
[data-testid="metric-container"] label { color: #94a3b8 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; }

.stDataFrame { background: #161b27 !important; border-radius: 10px; }

.hero {
    background: linear-gradient(135deg, #052e16 0%, #14532d 50%, #166534 100%);
    border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
    border: 1px solid #16a34a;
}
.hero h1 { font-size: 1.9rem; font-weight: 800; color: #f0fdf4; margin: 0; }
.hero p  { font-size: 0.95rem; color: #86efac; margin: 6px 0 0; }

.price-card {
    background: #161b27; border: 1px solid #1e2d40; border-radius: 14px;
    padding: 22px 18px; text-align: center; border-top: 4px solid #16a34a; margin-bottom: 12px;
}
.price-main  { font-size: 2.3rem; font-weight: 800; color: #4ade80; }
.price-label { font-size: 0.85rem; color: #64748b; margin-top: 4px; }
.price-change-up   { color: #f87171; font-weight: 600; font-size: 0.95rem; margin-top: 8px; }
.price-change-down { color: #4ade80; font-weight: 600; font-size: 0.95rem; margin-top: 8px; }

.section-title {
    font-size: 1.15rem; font-weight: 700; color: #f1f5f9;
    margin: 24px 0 14px; padding-bottom: 8px; border-bottom: 2px solid #16a34a;
}
.tip-box { background: #052e16; border: 1px solid #16a34a; border-radius: 10px; padding: 14px 18px; margin: 10px 0; font-size: 0.9rem; color: #86efac; }
.alert-high { background: #2d1515; border-left: 5px solid #f87171; border-radius: 10px; padding: 14px; margin: 8px 0; color: #fca5a5; }
.alert-low  { background: #052e16; border-left: 5px solid #4ade80; border-radius: 10px; padding: 14px; margin: 8px 0; color: #86efac; }
.alert-warn { background: #1c1208; border-left: 5px solid #fb923c; border-radius: 10px; padding: 14px; margin: 8px 0; color: #fdba74; }
.badge-live  { background:#052e16; color:#4ade80; padding:4px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.update-status { background: #161b27; border-radius: 10px; padding: 12px 16px; margin-bottom: 16px; border: 1px solid #1e2d40; font-size:0.88rem; color:#94a3b8; }
</style>
""", unsafe_allow_html=True)

# ── Init DB ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Setting up database — please wait...")
def boot():
    """
    Runs ONCE per server session (cached by Streamlit).
    1. Init DB tables
    2. Seed commodities + markets if empty
    3. Generate 120-day synthetic price history if empty
    Never runs again until the server restarts.
    """
    import numpy as np
    from datetime import date, timedelta

    init_db()

    # Check if already seeded using app_meta flag
    from database import get_meta, set_meta
    already_seeded = get_meta("seeded")
    if already_seeded == "true":
        return True

    # Seed reference data (commodities + markets)
    seed_reference_data()

    # Fast vectorized synthetic seed — 120 days of history
    from database import read_commodities, read_markets
    comms_df = read_commodities()
    mkts_df  = read_markets()

    BASE_PRICES = {
        "Rice":2200,"Wheat":2100,"Maize":1800,"Barley":1700,"Jowar":2000,"Bajra":1900,"Ragi":2300,
        "Tur Dal":8500,"Moong Dal":9500,"Urad Dal":9000,"Chana Dal":7000,"Masoor Dal":7500,
        "Tomato":1500,"Onion":1800,"Potato":1400,"Brinjal":1200,"Cabbage":900,"Cauliflower":1100,
        "Carrot":1600,"Spinach":1000,"Bitter Gourd":1800,"Capsicum":2200,
        "Banana":2000,"Mango":5000,"Apple":8000,"Grapes":4500,"Orange":3500,"Papaya":1800,
        "Turmeric":8000,"Chilli":12000,"Coriander":7000,"Cumin":18000,"Ginger":6000,"Garlic":12000,
        "Groundnut":5500,"Mustard":5200,"Soybean":4800,"Sunflower":5000,"Sesame":9000,
    }
    MARKET_PREMIUM = {
        "Azadpur":1.05,"Vashi":1.08,"Bowenpally":0.97,
        "Koyambedu":1.02,"Yeshwanthpur":1.06,"Gultekdi":1.04,
    }

    days_back = 120
    today     = date.today()
    dates     = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_back)]

    records = []
    rng = np.random.default_rng(42)
    for _, c in comms_df.iterrows():
        base = BASE_PRICES.get(c["name"], 2000)
        for _, m in mkts_df.iterrows():
            doys     = np.array([(today-timedelta(days=i)).timetuple().tm_yday for i in range(days_back)])
            seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * (doys - 90) / 365)
            noise    = rng.normal(0, 0.03, days_back)
            premium  = MARKET_PREMIUM.get(m["market_name"], 1.0)
            modal    = np.round(base * seasonal * premium * (1 + noise), 2)
            spread   = rng.uniform(0.03, 0.07, days_back)
            min_p    = np.round(modal * (1 - spread), 2)
            max_p    = np.round(modal * (1 + spread), 2)
            arrivals = np.round(rng.uniform(50, 500, days_back), 1)
            for i, d in enumerate(dates):
                records.append((
                    int(c["id"]), int(m["id"]), d,
                    float(min_p[i]), float(max_p[i]), float(modal[i]),
                    float(arrivals[i]), "Synthetic"
                ))

    from database import bulk_insert_prices
    bulk_insert_prices(records)

    # Mark as seeded so we never run this again
    set_meta("seeded", "true")
    set_meta("seed_date", str(date.today()))
    return True

boot()

# ── Load reference data ───────────────────────────────────────────────────────
commodities_df = read_commodities()
markets_df     = read_markets()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 Agri Price Tracker")
    st.markdown("**Daily crop prices from every mandi**")
    st.markdown("---")

    st.markdown("### 🔍 Select Crop")
    categories = sorted(commodities_df["category"].unique().tolist()) if not commodities_df.empty else []
    sel_cat = st.selectbox("Category", ["All"] + categories)

    if sel_cat != "All":
        comm_list = commodities_df[commodities_df["category"] == sel_cat]["name"].tolist()
    else:
        comm_list = commodities_df["name"].tolist()

    sel_commodity = st.selectbox("Commodity", comm_list) if comm_list else None

    st.markdown("### 📍 Select Market")
    states = sorted(markets_df["state"].unique().tolist()) if not markets_df.empty else []
    sel_state = st.selectbox("State", ["All"] + states)

    if sel_state != "All":
        mkt_list = markets_df[markets_df["state"] == sel_state]["market_name"].tolist()
    else:
        mkt_list = markets_df["market_name"].tolist()

    sel_market = st.selectbox("Market", mkt_list) if mkt_list else None

    st.markdown("---")
    st.markdown("### 📅 Date Range")
    days = st.select_slider(
        "Days to show",
        options=[7, 14, 30, 60, 90],
        value=30
    )

    st.markdown("---")
    if st.button("🔄 Refresh Prices", use_container_width=True):
        with st.spinner("Fetching latest prices..."):
            count = fetch_and_store_prices(days_back=2)
            st.success(f"✓ Updated {count} records")
        st.rerun()

    # Data freshness indicator
    stats = get_summary_stats()
    latest = stats.get("latest_date", "N/A")
    today  = date.today().isoformat()
    fresh  = "🟢 Today's data" if latest == today else f"🟡 Last: {latest}"
    st.markdown(f"**Data Status:** {fresh}")

# ── Resolve IDs ───────────────────────────────────────────────────────────────
def get_id(df, col, val):
    rows = df[df[col] == val]
    return int(rows["id"].iloc[0]) if not rows.empty else None

commodity_id = get_id(commodities_df, "name", sel_commodity) if sel_commodity else None
market_id    = get_id(markets_df, "market_name", sel_market) if sel_market else None

# ── HERO BANNER ───────────────────────────────────────────────────────────────
stats = get_summary_stats()
latest_date = stats.get("latest_date", "N/A")
is_today    = latest_date == date.today().isoformat()
data_badge  = "🟢 Live — Today's Data" if is_today else f"🟡 Last Updated: {latest_date}"

st.markdown(f"""
<div class="hero">
  <h1>🌾 Agri Price Tracker</h1>
  <p>Live commodity prices from Agmarknet &nbsp;·&nbsp; Anomaly detection &nbsp;·&nbsp; State analytics</p>
  <p style="margin-top:10px; font-size:0.85rem; opacity:0.8">{data_badge} &nbsp;|&nbsp;
     {stats['total_commodities']} Commodities &nbsp;|&nbsp;
     {stats['total_markets']} Mandis &nbsp;|&nbsp;
     {stats['total_states']} States</p>
</div>
""", unsafe_allow_html=True)

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Today's Price",
    "📈 Price Trend",
    "⚠️ Price Alerts",
    "🗺️ All States",
])

CHART_STYLE = dict(
    paper_bgcolor="#161b27",
    plot_bgcolor="#0f1117",
    font=dict(color="#94a3b8", size=12),
    xaxis=dict(gridcolor="#1e2d40", linecolor="#1e2d40"),
    yaxis=dict(gridcolor="#1e2d40", linecolor="#1e2d40"),
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — TODAY'S PRICE
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    if commodity_id and market_id:
        trend = get_price_trend(commodity_id, market_id, days=days)

        if not trend.empty:
            trend["price_date"] = pd.to_datetime(trend["price_date"])
            trend = trend.sort_values("price_date")

            latest_row  = trend.iloc[-1]
            prev_row    = trend.iloc[-2] if len(trend) > 1 else latest_row
            week_row    = trend.iloc[-7] if len(trend) >= 7 else trend.iloc[0]

            modal  = latest_row["modal_price"]
            min_p  = latest_row["min_price"]
            max_p  = latest_row["max_price"]
            chg_1d = modal - prev_row["modal_price"]
            chg_7d = modal - week_row["modal_price"]
            chg_1d_pct = chg_1d / prev_row["modal_price"] * 100
            chg_7d_pct = chg_7d / week_row["modal_price"] * 100

            arrow_1d = "▲" if chg_1d >= 0 else "▼"
            arrow_7d = "▲" if chg_7d >= 0 else "▼"
            cls_1d   = "price-change-up" if chg_1d >= 0 else "price-change-down"
            cls_7d   = "price-change-up" if chg_7d >= 0 else "price-change-down"

            # Big price cards
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="price-card">
                  <div class="price-label">Today's Price</div>
                  <div class="price-main">₹{modal:,.0f}</div>
                  <div class="price-label">per Quintal (100 kg)</div>
                  <div class="{cls_1d}" style="margin-top:8px">
                    {arrow_1d} ₹{abs(chg_1d):,.0f} ({chg_1d_pct:+.1f}%) from yesterday
                  </div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="price-card" style="border-top-color:#f57c00">
                  <div class="price-label">Minimum</div>
                  <div class="price-main" style="color:#e65100">₹{min_p:,.0f}</div>
                  <div class="price-label">Lowest price today</div>
                  <div style="margin-top:8px; color:#888; font-size:0.85rem">
                    ₹{trend['min_price'].min():,.0f} lowest in {days}d
                  </div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="price-card" style="border-top-color:#1565c0">
                  <div class="price-label">Maximum</div>
                  <div class="price-main" style="color:#1565c0">₹{max_p:,.0f}</div>
                  <div class="price-label">Highest price today</div>
                  <div style="margin-top:8px; color:#888; font-size:0.85rem">
                    ₹{trend['max_price'].max():,.0f} highest in {days}d
                  </div>
                </div>""", unsafe_allow_html=True)

            # Farmer tip
            avg_30 = trend["modal_price"].mean()
            if modal > avg_30 * 1.1:
                tip_msg = f"💡 <b>Good time to sell!</b> Today's price ₹{modal:,.0f} is {((modal/avg_30)-1)*100:.0f}% ABOVE the {days}-day average (₹{avg_30:,.0f}). Consider selling now."
                tip_cls = "alert-low"
            elif modal < avg_30 * 0.9:
                tip_msg = f"⚠️ <b>Prices are low.</b> Today's price ₹{modal:,.0f} is {((1-(modal/avg_30))*100):.0f}% BELOW the {days}-day average (₹{avg_30:,.0f}). Consider waiting if possible."
                tip_cls = "alert-high"
            else:
                tip_msg = f"📊 <b>Prices are normal.</b> Today's ₹{modal:,.0f} is near the {days}-day average (₹{avg_30:,.0f}). Monitor daily for changes."
                tip_cls = "tip-box"
            st.markdown(f'<div class="{tip_cls}">{tip_msg}</div>', unsafe_allow_html=True)

            # 7-day mini table
            st.markdown('<div class="section-title">📅 Last 7 Days</div>', unsafe_allow_html=True)
            last7 = trend.tail(7).copy().sort_values("price_date", ascending=False)
            last7["Date"]          = last7["price_date"].dt.strftime("%d %b %Y")
            last7["Modal (₹/Qt)"]  = last7["modal_price"].apply(lambda x: f"₹{x:,.0f}")
            last7["Min (₹/Qt)"]    = last7["min_price"].apply(lambda x: f"₹{x:,.0f}")
            last7["Max (₹/Qt)"]    = last7["max_price"].apply(lambda x: f"₹{x:,.0f}")
            last7["Change"]        = last7["modal_price"].diff(-1).apply(
                lambda x: f"▲ ₹{abs(x):,.0f}" if x > 0 else (f"▼ ₹{abs(x):,.0f}" if x < 0 else "—") if pd.notna(x) else "—"
            )
            st.dataframe(
                last7[["Date","Modal (₹/Qt)","Min (₹/Qt)","Max (₹/Qt)","Change"]],
                use_container_width=True, hide_index=True
            )

            # Arrivals & price today bar
            st.markdown(f'<div class="section-title">🚛 Market Arrivals — {sel_market}</div>', unsafe_allow_html=True)
            fig_arr = go.Figure()
            fig_arr.add_trace(go.Bar(
                x=trend["price_date"], y=trend["arrivals"],
                marker_color="#16a34a", name="Arrivals (Quintals)"
            ))
            fig_arr.update_layout(
                yaxis_title="Arrivals (Qt)", xaxis_title="Date",
                height=220, margin=dict(t=10, b=10), **CHART_STYLE
            )
            st.plotly_chart(fig_arr, use_container_width=True)

        else:
            st.info("No price data for this selection. Click 'Refresh' in the sidebar.")
    else:
        # Landing state — show top movers
        st.markdown('<div class="section-title">📊 Top Commodity Prices</div>', unsafe_allow_html=True)
        st.markdown("**👈 Select a commodity and market from the sidebar to see detailed prices.**")

        vol_df = get_top_commodities_by_volatility(12)
        if not vol_df.empty:
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.bar(
                    vol_df.sort_values("avg_price", ascending=True).head(10),
                    x="avg_price", y="commodity_name", orientation="h",
                    color="category", title="Average Prices (₹/Quintal)",
                    labels={"avg_price":"Price (₹)", "commodity_name":"Commodity"},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(height=350, **CHART_STYLE, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                fig2 = px.pie(
                    commodities_df.groupby("category").size().reset_index(name="count"),
                    values="count", names="category",
                    title="Commodities by Category",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hole=0.4,
                )
                fig2.update_layout(height=350, **CHART_STYLE)
                st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRICE TREND
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    if commodity_id and market_id:
        trend = get_price_trend(commodity_id, market_id, days=days)
        if not trend.empty:
            trend["price_date"] = pd.to_datetime(trend["price_date"])
            trend = trend.sort_values("price_date")
            trend["7d_avg"]  = trend["modal_price"].rolling(7,  min_periods=1).mean()
            trend["30d_avg"] = trend["modal_price"].rolling(30, min_periods=1).mean()

            st.markdown(f'<div class="section-title">📈 {sel_commodity} — {sel_market} Price Chart ({days} days)</div>', unsafe_allow_html=True)

            fig = go.Figure()
            # Range band
            fig.add_trace(go.Scatter(
                x=pd.concat([trend["price_date"], trend["price_date"][::-1]]),
                y=pd.concat([trend["max_price"], trend["min_price"][::-1]]),
                fill="toself", fillcolor="rgba(56,142,60,0.12)",
                line=dict(color="rgba(56,142,60,0)"),
                name="Min–Max Range", hoverinfo="skip"
            ))
            # Modal price
            fig.add_trace(go.Scatter(
                x=trend["price_date"], y=trend["modal_price"],
                mode="lines+markers", name="Modal Price",
                line=dict(color="#2e7d32", width=2.5),
                marker=dict(size=5, color="#2e7d32"),
            ))
            # 7d MA
            fig.add_trace(go.Scatter(
                x=trend["price_date"], y=trend["7d_avg"],
                mode="lines", name="7-Day Average",
                line=dict(color="#f57c00", width=2, dash="dash"),
            ))
            # 30d MA
            if days >= 30:
                fig.add_trace(go.Scatter(
                    x=trend["price_date"], y=trend["30d_avg"],
                    mode="lines", name="30-Day Average",
                    line=dict(color="#1565c0", width=1.5, dash="dot"),
                ))

            fig.update_layout(
                yaxis_title="Price (₹ per Quintal)",
                xaxis_title="Date",
                height=400,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                **CHART_STYLE
            )
            st.plotly_chart(fig, use_container_width=True)

            # Stats row
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Highest Price", f"₹{trend['max_price'].max():,.0f}")
            s2.metric("Lowest Price",  f"₹{trend['min_price'].min():,.0f}")
            avg = trend["modal_price"].mean()
            last = trend["modal_price"].iloc[-1]
            s3.metric("Average Price", f"₹{avg:,.0f}")
            s4.metric(f"Trend ({days}d)",
                      f"₹{last-trend['modal_price'].iloc[0]:+,.0f}",
                      f"{((last/trend['modal_price'].iloc[0])-1)*100:+.1f}%")

            # Compare across markets for same commodity
            st.markdown(f'<div class="section-title">🏪 {sel_commodity} — Price at Other Mandis</div>', unsafe_allow_html=True)
            all_mkt = read_prices(
                commodity_id=commodity_id,
                start_date=str(date.today() - timedelta(days=7))
            )
            if not all_mkt.empty:
                mkt_summary = all_mkt.groupby("market_name")["modal_price"].mean().reset_index()
                mkt_summary.columns = ["Mandi", "Avg Price (₹)"]
                mkt_summary = mkt_summary.sort_values("Avg Price (₹)", ascending=False)
                # Highlight selected market
                mkt_summary["You Selected"] = mkt_summary["Mandi"].apply(
                    lambda x: "⭐ " + x if x == sel_market else x
                )
                fig3 = px.bar(
                    mkt_summary, x="You Selected", y="Avg Price (₹)",
                    color="Avg Price (₹)",
                    color_continuous_scale=["#c8e6c9","#1b5e20"],
                    title=f"Where is {sel_commodity} priced highest? (Last 7 days)",
                    text="Avg Price (₹)",
                )
                fig3.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
                fig3.update_layout(height=350, **CHART_STYLE,
                                   coloraxis_showscale=False,
                                   xaxis_title="Mandi", xaxis_tickangle=-25)
                st.plotly_chart(fig3, use_container_width=True)

                best = mkt_summary.iloc[0]
                if best["Mandi"] != sel_market:
                    st.markdown(f"""
                    <div class="tip-box">
                    💡 <b>Better price available!</b> {best['Mandi']} currently offers ₹{best['Avg Price (₹)']:,.0f}
                    vs ₹{mkt_summary[mkt_summary['Mandi']==sel_market]['Avg Price (₹)'].values[0]:,.0f}
                    at {sel_market}. Consider selling there if transport cost allows.
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("No data found. Click Refresh in sidebar.")
    else:
        st.info("👈 Please select a commodity and market from the sidebar.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRICE ALERTS / ANOMALIES
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">⚠️ Price Alerts — Unusual Price Movements</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="tip-box">
    This page shows when prices suddenly jumped or dropped unusually.
    It helps you understand if there's a supply shortage, bumper crop, or market disruption.
    </div>""", unsafe_allow_html=True)

    col_detect, col_days2 = st.columns([1, 2])
    with col_detect:
        if st.button("🔍 Check for Alerts", type="primary", use_container_width=True):
            if commodity_id and market_id:
                with st.spinner("Scanning for unusual prices..."):
                    found = detect_anomalies(commodity_id, market_id)
                    st.success(f"Found {len(found)} unusual price movements")
            else:
                st.warning("Select a commodity and market first.")
    with col_days2:
        alert_days = st.slider("Show alerts from last N days", 7, 60, 30)

    anomalies_df = read_anomalies(alert_days)
    if not anomalies_df.empty:
        # Summary numbers
        a1, a2, a3 = st.columns(3)
        a1.metric("Total Alerts", len(anomalies_df))
        a2.metric("🔴 Critical / High",
                  len(anomalies_df[anomalies_df["severity"].isin(["Critical","High"])]))
        a3.metric("Commodities Affected", anomalies_df["commodity_name"].nunique())

        for _, row in anomalies_df.head(15).iterrows():
            icon = "🔴" if row["severity"] == "Critical" else \
                   "🟡" if row["severity"] == "High" else "🔵"
            cls  = "alert-warn" if row["severity"] in ("Critical","High") else "alert-high"
            dev  = row["deviation_pct"]
            direction = "⬆️ Price spike" if dev > 0 else "⬇️ Price drop"
            st.markdown(f"""
            <div class="{cls}">
              <b>{icon} {row['commodity_name']} @ {row['market_name']}, {row['state']}</b>
              &nbsp;&nbsp;<small>{row['detected_date']}</small><br>
              <b>{direction} of {abs(dev):.1f}%</b> — {row['description']}
            </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ No unusual price movements found in the last {} days.".format(alert_days))
        st.markdown("""
        <div class="tip-box">
        💡 Run the check above to scan for price spikes and drops in your selected commodity.
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ALL STATES ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🗺️ State-wise Price Comparison</div>',
                unsafe_allow_html=True)

    state_df = get_state_analytics()
    if not state_df.empty:
        # Which state has highest / lowest price
        best_state  = state_df.loc[state_df["avg_price"].idxmax()]
        worst_state = state_df.loc[state_df["avg_price"].idxmin()]

        b1, b2 = st.columns(2)
        b1.markdown(f"""
        <div class="price-card" style="border-top-color:#1565c0">
          <div class="price-label">💰 Highest Average Prices</div>
          <div class="price-main" style="color:#1565c0">{best_state['state']}</div>
          <div class="price-label">₹{best_state['avg_price']:,.0f}/Quintal average</div>
        </div>""", unsafe_allow_html=True)
        b2.markdown(f"""
        <div class="price-card" style="border-top-color:#388e3c">
          <div class="price-label">🌿 Lowest Average Prices</div>
          <div class="price-main" style="color:#388e3c">{worst_state['state']}</div>
          <div class="price-label">₹{worst_state['avg_price']:,.0f}/Quintal average</div>
        </div>""", unsafe_allow_html=True)

        fig = px.bar(
            state_df.sort_values("avg_price", ascending=True),
            x="avg_price", y="state", orientation="h",
            color="avg_price",
            color_continuous_scale=["#c8e6c9","#1b5e20"],
            text="avg_price",
            title="Average Commodity Price by State (₹/Quintal)",
            labels={"avg_price":"Avg Price (₹)","state":"State"},
        )
        fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        fig.update_layout(height=450, **CHART_STYLE, coloraxis_showscale=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        # Summary table — simple version
        st.markdown("**State Summary Table**")
        display = state_df[["state","num_markets","avg_price","max_price","min_price"]].copy()
        display.columns = ["State", "Markets", "Avg Price (₹)", "Highest (₹)", "Lowest (₹)"]
        for col in ["Avg Price (₹)","Highest (₹)","Lowest (₹)"]:
            display[col] = display[col].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(display.sort_values("Avg Price (₹)", ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No state data yet. Click Refresh in sidebar.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:40px; padding:16px; text-align:center;
            border-top:2px solid #c8e6c9; color:#4caf50; font-size:0.82rem; background:white; border-radius:10px;">
  🌾 <b>Agri Price Tracker</b> &nbsp;|&nbsp;
  Data from Agmarknet (data.gov.in) &nbsp;|&nbsp;
  Updated daily &nbsp;|&nbsp;
  AI Forecast powered by Machine Learning<br>
  <span style="color:#999; font-size:0.75rem;">
    Note: Forecasts are estimates. Always confirm prices at your local mandi before selling.
  </span>
</div>
""", unsafe_allow_html=True)
