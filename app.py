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
from ml_models import detect_anomalies
from ai_insights import generate_price_movement_insight
from database import bulk_insert_prices

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agri Price Tracker 🌾",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp {
    background: #0f1117;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2d40;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1e2d40 !important;
    border: 1px solid #2d4a6b !important;
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #16a34a !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #15803d !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161b27;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2d40;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #94a3b8;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 8px 18px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #16a34a !important;
    color: white !important;
    font-weight: 600 !important;
}

/* ── Dataframe ── */
.stDataFrame { background: #161b27; border-radius: 10px; }
iframe[title="st_aggrid"] { background: #161b27; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #161b27;
    border: 1px solid #1e2d40;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="metric-container"] label { color: #94a3b8 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; }

/* ── Custom cards ── */
.hero-card {
    background: linear-gradient(135deg, #052e16 0%, #14532d 50%, #166534 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    border: 1px solid #16a34a;
}
.hero-card h1 { font-size: 1.9rem; font-weight: 800; color: #f0fdf4; margin: 0; }
.hero-card p  { font-size: 0.95rem; color: #86efac; margin: 6px 0 0; }

.kpi-card {
    background: #161b27;
    border: 1px solid #1e2d40;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.kpi-val   { font-size: 2rem; font-weight: 700; color: #4ade80; }
.kpi-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase;
             letter-spacing: 0.8px; margin-top: 4px; }

.price-card {
    background: #161b27;
    border: 1px solid #1e2d40;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    border-top: 4px solid #16a34a;
    margin-bottom: 12px;
}
.price-main  { font-size: 2.3rem; font-weight: 800; color: #4ade80; }
.price-label { font-size: 0.85rem; color: #64748b; margin-top: 4px; }
.price-up    { color: #f87171; font-weight: 600; font-size: 0.95rem; margin-top: 8px; }
.price-down  { color: #4ade80; font-weight: 600; font-size: 0.95rem; margin-top: 8px; }

.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 24px 0 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid #16a34a;
}

.tip-box {
    background: #052e16;
    border: 1px solid #16a34a;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #86efac;
}
.alert-sell {
    background: #052e16;
    border-left: 4px solid #4ade80;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #86efac;
    font-size: 0.9rem;
}
.alert-wait {
    background: #2d1515;
    border-left: 4px solid #f87171;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #fca5a5;
    font-size: 0.9rem;
}
.alert-warn {
    background: #1c1208;
    border-left: 4px solid #fb923c;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #fdba74;
    font-size: 0.9rem;
}
.anomaly-card {
    background: #161b27;
    border: 1px solid #1e2d40;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Chart theme — dark ────────────────────────────────────────────────────────
CHART = dict(
    paper_bgcolor="#161b27",
    plot_bgcolor="#0f1117",
    font=dict(color="#94a3b8", size=12),
    xaxis=dict(gridcolor="#1e2d40", linecolor="#1e2d40", zerolinecolor="#1e2d40"),
    yaxis=dict(gridcolor="#1e2d40", linecolor="#1e2d40", zerolinecolor="#1e2d40"),
    legend=dict(bgcolor="#161b27", bordercolor="#1e2d40"),
)

# ── Boot / seed ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🌱 Loading market data — first run takes ~20s...")
def boot():
    import numpy as np
    from datetime import date, timedelta
    init_db()
    if get_meta("seeded") == "true":
        return True
    seed_reference_data()
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
    PREMIUM = {"Azadpur":1.05,"Vashi":1.08,"Bowenpally":0.97,"Koyambedu":1.02,"Yeshwanthpur":1.06,"Gultekdi":1.04}
    days_back = 120
    today = date.today()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_back)]
    records = []
    rng = np.random.default_rng(42)
    for _, c in comms_df.iterrows():
        base = BASE_PRICES.get(c["name"], 2000)
        for _, m in mkts_df.iterrows():
            doys     = np.array([(today-timedelta(days=i)).timetuple().tm_yday for i in range(days_back)])
            seasonal = 1.0 + 0.15*np.sin(2*np.pi*(doys-90)/365)
            noise    = rng.normal(0, 0.03, days_back)
            premium  = PREMIUM.get(m["market_name"], 1.0)
            modal    = np.round(base*seasonal*premium*(1+noise), 2)
            spread   = rng.uniform(0.03, 0.07, days_back)
            min_p    = np.round(modal*(1-spread), 2)
            max_p    = np.round(modal*(1+spread), 2)
            arrivals = np.round(rng.uniform(50, 500, days_back), 1)
            for i, d in enumerate(dates):
                records.append((int(c["id"]),int(m["id"]),d,
                                float(min_p[i]),float(max_p[i]),float(modal[i]),
                                float(arrivals[i]),"Synthetic"))
    bulk_insert_prices(records)
    set_meta("seeded","true")
    set_meta("seed_date", str(date.today()))
    return True

boot()

# ── Load data ─────────────────────────────────────────────────────────────────
commodities_df = read_commodities()
markets_df     = read_markets()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 10px; text-align:center;">
        <div style="font-size:2rem">🌾</div>
        <div style="font-size:1.1rem; font-weight:700; color:#4ade80;">Agri Price Tracker</div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">Live Commodity Prices</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("#### 🌿 Select Crop")
    categories = sorted(commodities_df["category"].unique().tolist()) if not commodities_df.empty else []
    sel_cat = st.selectbox("Category", ["All"] + categories, label_visibility="collapsed")
    comm_list = (commodities_df[commodities_df["category"]==sel_cat]["name"].tolist()
                 if sel_cat != "All" else commodities_df["name"].tolist())
    sel_commodity = st.selectbox("Commodity", comm_list, label_visibility="collapsed") if comm_list else None

    st.markdown("#### 📍 Select Market")
    states = sorted(markets_df["state"].unique().tolist()) if not markets_df.empty else []
    sel_state = st.selectbox("State", ["All"] + states, label_visibility="collapsed")
    mkt_list = (markets_df[markets_df["state"]==sel_state]["market_name"].tolist()
                if sel_state != "All" else markets_df["market_name"].tolist())
    sel_market = st.selectbox("Market", mkt_list, label_visibility="collapsed") if mkt_list else None

    st.divider()
    st.markdown("#### 📅 Date Range")
    days = st.select_slider("Days", options=[7,14,30,60,90], value=30, label_visibility="collapsed")

    st.divider()
    if st.button("🔄 Refresh Prices", use_container_width=True):
        with st.spinner("Fetching..."):
            count = fetch_and_store_prices(days_back=2)
            st.success(f"✓ {count} records updated")
        st.rerun()

    stats_s = get_summary_stats()
    latest  = stats_s.get("latest_date","N/A")
    is_fresh = latest == date.today().isoformat()
    st.markdown(
        f"<div style='text-align:center; margin-top:8px; font-size:0.8rem;'>"
        f"{'🟢 Live data today' if is_fresh else f'🟡 Last: {latest}'}</div>",
        unsafe_allow_html=True
    )

# ── IDs ───────────────────────────────────────────────────────────────────────
def get_id(df, col, val):
    rows = df[df[col] == val]
    return int(rows["id"].iloc[0]) if not rows.empty else None

commodity_id = get_id(commodities_df, "name", sel_commodity) if sel_commodity else None
market_id    = get_id(markets_df, "market_name", sel_market)  if sel_market    else None

# ── HERO ──────────────────────────────────────────────────────────────────────
stats = get_summary_stats()
latest_date = stats.get("latest_date","N/A")
badge = "🟢 Live — Today's Data" if latest_date == date.today().isoformat() else f"🟡 Last Updated: {latest_date}"

st.markdown(f"""
<div class="hero-card">
  <h1>🌾 Agri Price Tracker</h1>
  <p>Live commodity prices from Agmarknet &nbsp;·&nbsp; Anomaly detection &nbsp;·&nbsp; State analytics</p>
  <p style="margin-top:12px; font-size:0.82rem; color:#6ee7b7;">
    {badge} &nbsp;|&nbsp; {stats['total_commodities']} Commodities &nbsp;|&nbsp;
    {stats['total_markets']} Mandis &nbsp;|&nbsp; {stats['total_states']} States
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
def kpi(col, val, label):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-val">{val}</div>
      <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, f"{stats['total_records']:,}", "Price Records")
kpi(k2, stats['total_commodities'],    "Commodities")
kpi(k3, stats['total_markets'],        "Markets")
kpi(k4, stats['total_states'],         "States")
kpi(k5, stats['anomalies_30d'],        "Alerts 30d")
kpi(k6, latest_date,                   "Latest Data")

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS — 4 only (forecast removed) ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Today's Price",
    "📈 Price Trend",
    "⚠️ Price Alerts",
    "🗺️ All States",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — TODAY'S PRICE
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    if commodity_id and market_id:
        trend = get_price_trend(commodity_id, market_id, days=days)
        if not trend.empty:
            trend["price_date"] = pd.to_datetime(trend["price_date"])
            trend = trend.sort_values("price_date")

            latest_row = trend.iloc[-1]
            prev_row   = trend.iloc[-2] if len(trend) > 1 else latest_row
            week_row   = trend.iloc[-7] if len(trend) >= 7 else trend.iloc[0]

            modal  = latest_row["modal_price"]
            min_p  = latest_row["min_price"]
            max_p  = latest_row["max_price"]
            chg_1d = modal - prev_row["modal_price"]
            chg_7d = modal - week_row["modal_price"]
            chg_pct_1d = chg_1d / prev_row["modal_price"] * 100
            arrow_1d = "▲" if chg_1d >= 0 else "▼"
            cls_1d   = "price-up" if chg_1d >= 0 else "price-down"

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="price-card">
                  <div class="price-label">Today's Modal Price</div>
                  <div class="price-main">₹{modal:,.0f}</div>
                  <div class="price-label">per Quintal (100 kg)</div>
                  <div class="{cls_1d}">{arrow_1d} ₹{abs(chg_1d):,.0f} ({chg_pct_1d:+.1f}%) from yesterday</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="price-card" style="border-top-color:#f97316">
                  <div class="price-label">Minimum Price</div>
                  <div class="price-main" style="color:#fb923c">₹{min_p:,.0f}</div>
                  <div class="price-label">Lowest today</div>
                  <div style="color:#64748b; font-size:0.82rem; margin-top:8px">
                    ₹{trend['min_price'].min():,.0f} lowest in {days}d
                  </div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="price-card" style="border-top-color:#60a5fa">
                  <div class="price-label">Maximum Price</div>
                  <div class="price-main" style="color:#60a5fa">₹{max_p:,.0f}</div>
                  <div class="price-label">Highest today</div>
                  <div style="color:#64748b; font-size:0.82rem; margin-top:8px">
                    ₹{trend['max_price'].max():,.0f} highest in {days}d
                  </div>
                </div>""", unsafe_allow_html=True)

            # Sell tip
            avg_30 = trend["modal_price"].mean()
            if modal > avg_30 * 1.1:
                st.markdown(f"""<div class="alert-sell">
                💡 <b>Good time to sell!</b> Today ₹{modal:,.0f} is
                {((modal/avg_30)-1)*100:.0f}% ABOVE the {days}-day average (₹{avg_30:,.0f}).
                </div>""", unsafe_allow_html=True)
            elif modal < avg_30 * 0.9:
                st.markdown(f"""<div class="alert-wait">
                ⚠️ <b>Prices are low.</b> Today ₹{modal:,.0f} is
                {((1-modal/avg_30)*100):.0f}% BELOW the {days}-day average (₹{avg_30:,.0f}).
                Consider waiting if you can store.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="tip-box">
                📊 <b>Prices are stable.</b> Today ₹{modal:,.0f} is near the
                {days}-day average (₹{avg_30:,.0f}).
                </div>""", unsafe_allow_html=True)

            # Last 7 days table
            st.markdown('<div class="section-title">📅 Last 7 Days</div>', unsafe_allow_html=True)
            last7 = trend.tail(7).copy().sort_values("price_date", ascending=False)
            last7["Date"]         = last7["price_date"].dt.strftime("%d %b %Y")
            last7["Modal (₹/Qt)"] = last7["modal_price"].apply(lambda x: f"₹{x:,.0f}")
            last7["Min (₹/Qt)"]   = last7["min_price"].apply(lambda x: f"₹{x:,.0f}")
            last7["Max (₹/Qt)"]   = last7["max_price"].apply(lambda x: f"₹{x:,.0f}")
            last7["Change"]       = last7["modal_price"].diff(-1).apply(
                lambda x: f"▲ ₹{abs(x):,.0f}" if x > 0 else (f"▼ ₹{abs(x):,.0f}" if x < 0 else "—") if pd.notna(x) else "—"
            )
            st.dataframe(last7[["Date","Modal (₹/Qt)","Min (₹/Qt)","Max (₹/Qt)","Change"]],
                         use_container_width=True, hide_index=True)

            # Arrivals chart
            st.markdown(f'<div class="section-title">🚛 Market Arrivals — {sel_market}</div>',
                        unsafe_allow_html=True)
            fig_arr = go.Figure(go.Bar(
                x=trend["price_date"], y=trend["arrivals"],
                marker_color="#16a34a", marker_line_width=0,
                name="Arrivals (Qt)"
            ))
            fig_arr.update_layout(
                yaxis_title="Arrivals (Quintals)", xaxis_title="Date",
                height=240, margin=dict(t=10, b=10), **CHART
            )
            st.plotly_chart(fig_arr, use_container_width=True)

        else:
            st.info("No data for this selection. Click 🔄 Refresh in the sidebar.")
    else:
        st.markdown('<div class="section-title">📊 Overview — All Commodities</div>',
                    unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b'>👈 Select a commodity and market from the sidebar.</p>",
                    unsafe_allow_html=True)
        vol_df = get_top_commodities_by_volatility(12)
        if not vol_df.empty:
            c_a, c_b = st.columns(2)
            with c_a:
                fig = px.bar(vol_df.sort_values("avg_price", ascending=True).head(10),
                             x="avg_price", y="commodity_name", orientation="h",
                             color="category", title="Average Prices (₹/Quintal)",
                             labels={"avg_price":"Price (₹)","commodity_name":""},
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(height=360, **CHART, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c_b:
                fig2 = px.pie(commodities_df.groupby("category").size().reset_index(name="n"),
                              values="n", names="category", title="Commodities by Category",
                              color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.45)
                fig2.update_layout(height=360, **CHART)
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

            st.markdown(f'<div class="section-title">📈 {sel_commodity} — {sel_market} ({days} days)</div>',
                        unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.concat([trend["price_date"], trend["price_date"][::-1]]),
                y=pd.concat([trend["max_price"], trend["min_price"][::-1]]),
                fill="toself", fillcolor="rgba(22,163,74,0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Min–Max Range", hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=trend["price_date"], y=trend["modal_price"],
                mode="lines+markers", name="Modal Price",
                line=dict(color="#4ade80", width=2.5),
                marker=dict(size=4, color="#4ade80"),
            ))
            fig.add_trace(go.Scatter(
                x=trend["price_date"], y=trend["7d_avg"],
                mode="lines", name="7-Day Avg",
                line=dict(color="#fb923c", width=2, dash="dash"),
            ))
            if days >= 30:
                fig.add_trace(go.Scatter(
                    x=trend["price_date"], y=trend["30d_avg"],
                    mode="lines", name="30-Day Avg",
                    line=dict(color="#60a5fa", width=1.5, dash="dot"),
                ))
            fig.update_layout(
                yaxis_title="Price (₹/Quintal)", height=420,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                **CHART
            )
            st.plotly_chart(fig, use_container_width=True)

            s1,s2,s3,s4 = st.columns(4)
            s1.metric("📈 Highest", f"₹{trend['max_price'].max():,.0f}")
            s2.metric("📉 Lowest",  f"₹{trend['min_price'].min():,.0f}")
            s3.metric("📊 Average", f"₹{trend['modal_price'].mean():,.0f}")
            last_p = trend["modal_price"].iloc[-1]
            first_p = trend["modal_price"].iloc[0]
            s4.metric(f"📅 {days}d Change",
                      f"₹{last_p-first_p:+,.0f}",
                      f"{((last_p/first_p)-1)*100:+.1f}%")

            # Cross-market comparison
            st.markdown(f'<div class="section-title">🏪 {sel_commodity} — Compare Mandis (last 7 days)</div>',
                        unsafe_allow_html=True)
            all_mkt = read_prices(commodity_id=commodity_id,
                                  start_date=str(date.today()-timedelta(days=7)))
            if not all_mkt.empty:
                mkt_avg = all_mkt.groupby("market_name")["modal_price"].mean().reset_index()
                mkt_avg.columns = ["Mandi","Avg Price (₹)"]
                mkt_avg = mkt_avg.sort_values("Avg Price (₹)", ascending=False)
                mkt_avg["Label"] = mkt_avg["Mandi"].apply(lambda x: "⭐ "+x if x==sel_market else x)
                fig3 = px.bar(mkt_avg, x="Label", y="Avg Price (₹)",
                              color="Avg Price (₹)",
                              color_continuous_scale=["#14532d","#4ade80"],
                              text="Avg Price (₹)",
                              title=f"Best price for {sel_commodity} across mandis")
                fig3.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside",
                                   marker_line_width=0)
                fig3.update_layout(height=360, **CHART,
                                   coloraxis_showscale=False, xaxis_tickangle=-25)
                st.plotly_chart(fig3, use_container_width=True)

                best = mkt_avg.iloc[0]
                if best["Mandi"] != sel_market:
                    curr_val = mkt_avg[mkt_avg["Mandi"]==sel_market]["Avg Price (₹)"].values
                    if len(curr_val):
                        st.markdown(f"""<div class="tip-box">
                        💡 <b>Better price at {best['Mandi']}:</b> ₹{best['Avg Price (₹)']:,.0f}
                        vs ₹{curr_val[0]:,.0f} at {sel_market}.
                        Consider selling there if transport cost allows.
                        </div>""", unsafe_allow_html=True)
        else:
            st.info("No data. Click 🔄 Refresh in sidebar.")
    else:
        st.info("👈 Select a commodity and market from the sidebar.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRICE ALERTS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">⚠️ Unusual Price Movements</div>', unsafe_allow_html=True)
    st.markdown("""<div class="tip-box">
    Detects sudden price spikes and drops — may signal supply shortages,
    bumper harvests, or market disruptions.
    </div>""", unsafe_allow_html=True)

    col_det, col_sl = st.columns([1, 2])
    with col_det:
        if st.button("🔍 Scan for Alerts", type="primary", use_container_width=True):
            if commodity_id and market_id:
                with st.spinner("Scanning..."):
                    found = detect_anomalies(commodity_id, market_id)
                    st.success(f"Found {len(found)} unusual movements")
            else:
                st.warning("Select a commodity and market first.")
    with col_sl:
        alert_days = st.slider("Days to look back", 7, 60, 30, label_visibility="visible")

    anomalies_df = read_anomalies(alert_days)
    if not anomalies_df.empty:
        a1,a2,a3 = st.columns(3)
        a1.metric("Total Alerts", len(anomalies_df))
        a2.metric("🔴 Critical/High",
                  len(anomalies_df[anomalies_df["severity"].isin(["Critical","High"])]))
        a3.metric("Commodities Affected", anomalies_df["commodity_name"].nunique())

        st.markdown("<br>", unsafe_allow_html=True)
        for _, row in anomalies_df.head(15).iterrows():
            icon = "🔴" if row["severity"]=="Critical" else "🟡" if row["severity"]=="High" else "🔵"
            dev  = row["deviation_pct"]
            direction = "⬆️ Spike" if dev > 0 else "⬇️ Drop"
            cls  = "alert-wait" if row["severity"] in ("Critical","High") else "alert-warn"
            st.markdown(f"""
            <div class="{cls}">
              <b>{icon} {row['commodity_name']} @ {row['market_name']}, {row['state']}</b>
              &nbsp;<small style="opacity:0.7">{row['detected_date']}</small><br>
              <b>{direction} {abs(dev):.1f}%</b> — {row['description']}
            </div>""", unsafe_allow_html=True)
    else:
        st.success(f"✅ No unusual movements in the last {alert_days} days.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ALL STATES
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🗺️ State-wise Price Comparison</div>', unsafe_allow_html=True)
    state_df = get_state_analytics()
    if not state_df.empty:
        best_s  = state_df.loc[state_df["avg_price"].idxmax()]
        worst_s = state_df.loc[state_df["avg_price"].idxmin()]
        b1,b2 = st.columns(2)
        b1.markdown(f"""
        <div class="price-card" style="border-top-color:#60a5fa">
          <div class="price-label">💰 Highest Avg Prices</div>
          <div class="price-main" style="color:#60a5fa">{best_s['state']}</div>
          <div class="price-label">₹{best_s['avg_price']:,.0f} / Quintal avg</div>
        </div>""", unsafe_allow_html=True)
        b2.markdown(f"""
        <div class="price-card" style="border-top-color:#4ade80">
          <div class="price-label">🌿 Lowest Avg Prices</div>
          <div class="price-main" style="color:#4ade80">{worst_s['state']}</div>
          <div class="price-label">₹{worst_s['avg_price']:,.0f} / Quintal avg</div>
        </div>""", unsafe_allow_html=True)

        fig = px.bar(state_df.sort_values("avg_price", ascending=True),
                     x="avg_price", y="state", orientation="h",
                     color="avg_price",
                     color_continuous_scale=["#14532d","#4ade80"],
                     text="avg_price",
                     title="Average Commodity Price by State (₹/Quintal)",
                     labels={"avg_price":"Avg Price (₹)","state":""})
        fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside",
                          marker_line_width=0)
        fig.update_layout(height=480, **CHART, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**State Summary**")
        disp = state_df[["state","num_markets","avg_price","max_price","min_price"]].copy()
        disp.columns = ["State","Markets","Avg (₹)","High (₹)","Low (₹)"]
        for col in ["Avg (₹)","High (₹)","Low (₹)"]:
            disp[col] = disp[col].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(disp.sort_values("Avg (₹)", ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No data yet. Click 🔄 Refresh in sidebar.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px; padding:20px; text-align:center;
            border-top:1px solid #1e2d40; color:#475569; font-size:0.8rem;">
  🌾 <b style="color:#4ade80">Agri Price Tracker</b> &nbsp;·&nbsp;
  Data from Agmarknet (data.gov.in) &nbsp;·&nbsp; Updated daily &nbsp;·&nbsp;
  ML anomaly detection<br>
  <span style="color:#334155; font-size:0.72rem;">
    Prices shown are indicative. Always verify at your local mandi before selling.
  </span>
</div>
""", unsafe_allow_html=True)
