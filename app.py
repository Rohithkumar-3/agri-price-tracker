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
    get_price_trend, read_scheduler_log,
)
from data_pipeline import fetch_and_store_prices, seed_reference_data
from ml_models import forecast_prices, detect_anomalies
from ai_insights import generate_price_movement_insight



# Rest of your app

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agri Price Tracker 🌾",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("Agri Price Tracker")

st.markdown("""
Track daily commodity prices across Indian markets.
View trends, forecasts, anomalies, and market insights to help farmers and traders make informed decisions.
""")

# ── CSS — clean, large text, green farm theme ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp { background: #f0f7f0; }

/* Top hero banner */
.hero {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    color: white;
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0; }
.hero p  { font-size: 1rem; margin: 6px 0 0; opacity: 0.9; }

/* Big price card */
.price-card {
    background: white;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-top: 5px solid #2e7d32;
    margin-bottom: 12px;
}
.price-main  { font-size: 2.4rem; font-weight: 700; color: #1b5e20; }
.price-label { font-size: 0.9rem; color: #666; margin-top: 4px; }
.price-change-up   { color: #d32f2f; font-weight: 600; font-size: 1rem; }
.price-change-down { color: #388e3c; font-weight: 600; font-size: 1rem; }

/* Alert cards */
.alert-high { background:#fff3e0; border-left:5px solid #f57c00; border-radius:10px; padding:14px; margin:8px 0; }
.alert-low  { background:#e8f5e9; border-left:5px solid #388e3c; border-radius:10px; padding:14px; margin:8px 0; }
.alert-warn { background:#fce4ec; border-left:5px solid #c62828; border-radius:10px; padding:14px; margin:8px 0; }

/* Tip box */
.tip-box {
    background: #e8f5e9;
    border: 1px solid #a5d6a7;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.92rem;
    color: #1b5e20;
}

/* Section titles */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1b5e20;
    margin: 20px 0 12px;
    padding-bottom: 6px;
    border-bottom: 3px solid #a5d6a7;
}

/* Status badge */
.badge-live   { background:#e8f5e9; color:#2e7d32; padding:4px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-synth  { background:#fff8e1; color:#f57f17; padding:4px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-new    { background:#e3f2fd; color:#1565c0; padding:4px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #1b5e20 !important; }
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div { background: #2e7d32 !important; color: white !important; }

/* Forecast table */
.forecast-row { padding: 10px; border-bottom: 1px solid #e0e0e0; }
.forecast-date { font-weight:600; color:#333; }
.forecast-price { font-size:1.1rem; font-weight:700; color:#1b5e20; }
.forecast-range { font-size:0.82rem; color:#888; }

/* Update status */
.update-status {
    background: white;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 16px;
    border: 1px solid #c8e6c9;
    font-size:0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ── Init DB ───────────────────────────────────────────────────────────────────
@st.cache_resource
def boot():
    init_db()
    stats = get_summary_stats()
    if stats["total_records"] == 0:
        seed_reference_data()
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
  <p>Live commodity prices from Agmarknet · ML forecasting · Daily alerts</p>
  <p style="margin-top:10px; font-size:0.85rem; opacity:0.8">{data_badge} &nbsp;|&nbsp;
     {stats['total_commodities']} Commodities &nbsp;|&nbsp;
     {stats['total_markets']} Mandis &nbsp;|&nbsp;
     {stats['total_states']} States</p>
</div>
""", unsafe_allow_html=True)

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Today's Price",
    "📈 Price Trend",
    "🔮 Price Forecast",
    "⚠️ Price Alerts",
    "🗺️ All States",
])

CHART_STYLE = dict(
    paper_bgcolor="white", plot_bgcolor="#f9fbe7",
    font=dict(color="#1b5e20", size=13),
    xaxis=dict(gridcolor="#e8f5e9"),
    yaxis=dict(gridcolor="#e8f5e9"),
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
                marker_color="#81c784", name="Arrivals (Quintals)"
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
# TAB 3 — PRICE FORECAST
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    if commodity_id and market_id:
        st.markdown(f'<div class="section-title">🔮 Price Forecast — {sel_commodity} @ {sel_market}</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="tip-box">
        🤖 <b>How this works:</b> Our AI studies the last 90+ days of price data and predicts
        what price is likely in the next 14 days. This is an <i>estimate</i> — actual prices
        depend on weather, government policy, and other factors.
        </div>""", unsafe_allow_html=True)

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            run_fc = st.button("▶ Generate Forecast", type="primary", use_container_width=True)

        if run_fc or "fc_commodity" in st.session_state:
            if run_fc:
                with st.spinner("AI is analysing prices..."):
                    fc_df = forecast_prices(commodity_id, market_id, horizon=14)
                    st.session_state["fc_df"]        = fc_df
                    st.session_state["fc_commodity"] = sel_commodity
                    st.session_state["fc_market"]    = sel_market
            else:
                fc_df = st.session_state.get("fc_df", pd.DataFrame())

            if isinstance(fc_df, pd.DataFrame) and not fc_df.empty:
                hist = get_price_trend(commodity_id, market_id, days=60)
                hist["price_date"]       = pd.to_datetime(hist["price_date"])
                fc_df["forecast_date"]   = pd.to_datetime(fc_df["forecast_date"])

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist["price_date"], y=hist["modal_price"],
                    mode="lines", name="Past Prices",
                    line=dict(color="#2e7d32", width=2.5)
                ))
                if "lower_bound" in fc_df.columns:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([fc_df["forecast_date"], fc_df["forecast_date"][::-1]]),
                        y=pd.concat([fc_df["upper_bound"], fc_df["lower_bound"][::-1]]),
                        fill="toself", fillcolor="rgba(245,124,0,0.12)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="Likely Range", hoverinfo="skip"
                    ))
                fig.add_trace(go.Scatter(
                    x=fc_df["forecast_date"], y=fc_df["predicted_price"],
                    mode="lines+markers", name="AI Forecast",
                    line=dict(color="#f57c00", width=2.5, dash="dot"),
                    marker=dict(symbol="diamond", size=8, color="#f57c00"),
                ))
                fig.add_vline(
                    x=hist["price_date"].max(),
                    line_dash="dash", line_color="#999",
                    annotation_text="Today", annotation_position="top"
                )
                fig.update_layout(
                    yaxis_title="Price (₹/Quintal)", xaxis_title="Date",
                    height=380, hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    **CHART_STYLE
                )
                st.plotly_chart(fig, use_container_width=True)

                # Simple forecast card table
                st.markdown("**📋 Day-by-day Forecast**")
                cols = st.columns(7)
                for i, (_, row) in enumerate(fc_df.head(7).iterrows()):
                    with cols[i % 7]:
                        day_label = pd.to_datetime(row["forecast_date"]).strftime("%d %b")
                        pred      = row["predicted_price"]
                        curr      = hist["modal_price"].iloc[-1]
                        diff      = pred - curr
                        arrow     = "▲" if diff >= 0 else "▼"
                        clr       = "#d32f2f" if diff >= 0 else "#2e7d32"
                        st.markdown(f"""
                        <div class="price-card" style="padding:12px 8px">
                          <div style="font-size:0.78rem; color:#666">{day_label}</div>
                          <div style="font-size:1.25rem; font-weight:700; color:#1b5e20">₹{pred:,.0f}</div>
                          <div style="font-size:0.78rem; color:{clr}">{arrow} ₹{abs(diff):,.0f}</div>
                        </div>""", unsafe_allow_html=True)

                # Sell advice
                next7_avg = fc_df.head(7)["predicted_price"].mean()
                next14_avg = fc_df["predicted_price"].mean()
                curr_price = hist["modal_price"].iloc[-1]
                if next7_avg > curr_price * 1.05:
                    st.markdown("""
                    <div class="alert-low">
                    💰 <b>Forecast Tip:</b> Prices are expected to RISE in the next 7 days.
                    If you can wait, you may get a better price soon.
                    </div>""", unsafe_allow_html=True)
                elif next7_avg < curr_price * 0.95:
                    st.markdown("""
                    <div class="alert-high">
                    ⚠️ <b>Forecast Tip:</b> Prices may FALL in the next 7 days.
                    Consider selling sooner rather than later.
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="tip-box">
                    📊 <b>Forecast Tip:</b> Prices are expected to remain stable.
                    Sell when convenient.
                    </div>""", unsafe_allow_html=True)
            else:
                st.warning("Not enough data for forecast. Need at least 15 days of history.")
    else:
        st.info("👈 Select a commodity and market to see AI price forecast.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRICE ALERTS / ANOMALIES
# ════════════════════════════════════════════════════════════════════════════
with tab4:
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
# TAB 5 — ALL STATES ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab5:
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
