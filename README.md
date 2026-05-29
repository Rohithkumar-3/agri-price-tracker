# 🌾 Agri Price Tracker

> **Daily commodity price tracker for Indian agricultural markets — powered by live Agmarknet data, ML forecasting, and AI insights.**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data: Agmarknet](https://img.shields.io/badge/Data-Agmarknet%20API-orange)](https://data.gov.in)

---

## 📌 What It Does

| Feature | Description |
|---------|-------------|
| 💰 Today's Price | Shows min, max, modal price for any crop at any mandi |
| 📈 Price Trend | 7–90 day charts with 7-day and 30-day moving averages |
| 🔮 AI Forecast | 14-day price prediction using Random Forest ML |
| ⚠️ Price Alerts | Detects sudden price spikes and drops automatically |
| 🗺️ State Analytics | Compare prices across all Indian states |
| 🤖 Sell Advice | Plain-English tips: "Good time to sell" or "Wait" |

---

## 🗂️ Project Structure

```
agri-price-tracker/
├── app.py                  # Main Streamlit dashboard (farmer UI)
├── database.py             # SQLite CRUD layer
├── data_pipeline.py        # Fetch + clean + store prices (Agmarknet API)
├── ml_models.py            # Price forecasting + anomaly detection
├── ai_insights.py          # AI-generated market analysis
├── api.py                  # FastAPI REST endpoints
├── smart_scheduler.py      # 24/7 smart daily data fetcher
├── fast_seed.py            # One-time historical data seeder
├── test_api_connection.py  # Test your Agmarknet API key
├── requirements.txt
├── .env.example            # API key template
├── .streamlit/
│   └── config.toml         # Streamlit theme config
└── data/
    └── commodity_market.db # SQLite database (auto-created, gitignored)
```

---

## 🚀 Complete Setup Guide

### Prerequisites
- Python 3.10 or higher
- Git installed
- Agmarknet API key (free from data.gov.in)

---

### STEP 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/agri-price-tracker.git
cd agri-price-tracker
```

---

### STEP 2 — Create virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

---

### STEP 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### STEP 4 — Add your API key

```bash
# Copy the example file
cp .env.example .env

# Open .env and replace with your real key:
# AGMARKNET_API_KEY=your_actual_key_here
# ANTHROPIC_API_KEY=your_actual_key_here   ← optional, for AI insights
```

**To get Agmarknet API key:**
1. Go to https://data.gov.in
2. Click Login / Register (free)
3. Go to My Account → Generate API Key
4. Copy and paste into `.env`

**Test your key:**
```bash
python test_api_connection.py
```
You should see ✓ Connected! with sample records.

---

### STEP 5 — Seed historical data (run once)

```bash
python fast_seed.py
```

This creates 120 days of historical price data so the ML forecasting model has enough data to train on. Takes ~15 seconds.

---

### STEP 6 — Run the dashboard

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

### STEP 7 — (Optional) Run the REST API

```bash
# In a separate terminal:
uvicorn api:app --reload --port 8000
```

Open API docs at: **http://localhost:8000/docs**

---

### STEP 8 — (Optional) Start the daily scheduler

```bash
# In a separate terminal:
python smart_scheduler.py
```

This runs 24/7 and:
- Checks every hour between 7am–10pm IST if Agmarknet has new data
- Detects whether today's date has been uploaded (they sometimes upload late)
- Falls back to last known date if today's data isn't ready yet
- Automatically runs ML forecasts and anomaly detection after each fetch

---

## ☁️ Deploy to Streamlit Cloud (Free Hosting — 24/7)

### STEP 1 — Push to GitHub (see section below)

### STEP 2 — Go to https://share.streamlit.io

### STEP 3 — Click "New app"

### STEP 4 — Fill in:
- Repository: `your-username/agri-price-tracker`
- Branch: `main`
- Main file: `app.py`

### STEP 5 — Add secrets (instead of .env file)

In Streamlit Cloud → App Settings → Secrets, add:
```toml
AGMARKNET_API_KEY = "your_actual_key_here"
ANTHROPIC_API_KEY = "your_actual_key_here"
```

### STEP 6 — Click Deploy 🚀

Your app will be live at: `https://your-username-agri-price-tracker.streamlit.app`

---

## 📤 GitHub Push — Step by Step

### First time (new repo):

```bash
# 1. Initialize git in your project folder
cd agri-price-tracker
git init

# 2. Add all files (respects .gitignore — won't add .env or .db)
git add .

# 3. First commit
git commit -m "Initial commit: Agri Price Tracker"

# 4. Create repo on GitHub.com → New Repository → name: agri-price-tracker → Create
# (Do NOT add README or .gitignore from GitHub — we already have them)

# 5. Link and push
git remote add origin https://github.com/YOUR_USERNAME/agri-price-tracker.git
git branch -M main
git push -u origin main
```

### After making changes:

```bash
git add .
git commit -m "describe what you changed"
git push
```

### Useful git commands:

```bash
git status              # see what files changed
git log --oneline       # see commit history
git diff                # see exact changes
```

---

## 🔄 How the Smart Scheduler Works

```
Every hour (7am–10pm IST):
  ↓
Check Agmarknet API: what is the latest available date?
  ↓
Compare with our database: do we already have this date?
  ├── YES (same date) → Skip, retry in 1 hour
  │     (Agmarknet hasn't uploaded today's data yet)
  └── NO (new date found) → Run full pipeline:
        1. Fetch prices for new date
        2. Clean and store in SQLite
        3. Run anomaly detection
        4. Update ML forecasts (14 days ahead)
        5. Generate AI insights
        6. Log result
```

---

## 📡 API Endpoints

```
GET  /api/summary              KPI metrics
GET  /api/prices               Query prices with filters
GET  /api/forecast             14-day ML forecast
GET  /api/anomalies            Recent price alerts
GET  /api/analytics/state      State-wise analytics
POST /api/etl/run              Trigger manual data refresh
GET  /api/scheduler/log        Job history
```

Full docs: http://localhost:8000/docs

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| API key not working | Run `python test_api_connection.py YOUR_KEY` |
| No data showing | Run `python fast_seed.py` then restart app |
| Streamlit Cloud deploy fails | Check Secrets are added correctly in App Settings |
| `.env` file not found | Copy `.env.example` to `.env` and fill in keys |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👨‍💻 Author

Built with ❤️ for Indian farmers.
Data source: [Agmarknet via data.gov.in](https://data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070)
