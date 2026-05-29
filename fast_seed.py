"""
One-time seed script — run this ONCE to populate 120 days of historical data.
Usage: python fast_seed.py
"""
import numpy as np
import sqlite3
from datetime import date, timedelta
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from database import DB_PATH, init_db, create_commodity, create_market, read_commodities, read_markets

COMMODITY_CATALOG = {
    "Cereals":    ["Rice","Wheat","Maize","Barley","Jowar","Bajra","Ragi"],
    "Pulses":     ["Tur Dal","Moong Dal","Urad Dal","Chana Dal","Masoor Dal"],
    "Vegetables": ["Tomato","Onion","Potato","Brinjal","Cabbage","Cauliflower","Carrot","Spinach","Bitter Gourd","Capsicum"],
    "Fruits":     ["Banana","Mango","Apple","Grapes","Orange","Papaya"],
    "Spices":     ["Turmeric","Chilli","Coriander","Cumin","Ginger","Garlic"],
    "Oilseeds":   ["Groundnut","Mustard","Soybean","Sunflower","Sesame"],
}
MARKETS = [
    ("Azadpur","Delhi","North West Delhi"),
    ("Vashi","Maharashtra","Navi Mumbai"),
    ("Bowenpally","Telangana","Hyderabad"),
    ("Koyambedu","Tamil Nadu","Chennai"),
    ("Yeshwanthpur","Karnataka","Bengaluru"),
    ("Gultekdi","Maharashtra","Pune"),
    ("Ahmedabad Main","Gujarat","Ahmedabad"),
    ("Kolkata Main","West Bengal","Kolkata"),
    ("Patna Main","Bihar","Patna"),
    ("Lucknow Main","Uttar Pradesh","Lucknow"),
    ("Jaipur Sabzi Mandi","Rajasthan","Jaipur"),
    ("Bhopal Main","Madhya Pradesh","Bhopal"),
]
BASE_PRICES = {
    "Rice":2200,"Wheat":2100,"Maize":1800,"Barley":1700,"Jowar":2000,"Bajra":1900,"Ragi":2300,
    "Tur Dal":8500,"Moong Dal":9500,"Urad Dal":9000,"Chana Dal":7000,"Masoor Dal":7500,
    "Tomato":1500,"Onion":1800,"Potato":1400,"Brinjal":1200,"Cabbage":900,"Cauliflower":1100,
    "Carrot":1600,"Spinach":1000,"Bitter Gourd":1800,"Capsicum":2200,
    "Banana":2000,"Mango":5000,"Apple":8000,"Grapes":4500,"Orange":3500,"Papaya":1800,
    "Turmeric":8000,"Chilli":12000,"Coriander":7000,"Cumin":18000,"Ginger":6000,"Garlic":12000,
    "Groundnut":5500,"Mustard":5200,"Soybean":4800,"Sunflower":5000,"Sesame":9000,
}

print("Initialising database...")
init_db()
for cat, comms in COMMODITY_CATALOG.items():
    for c in comms: create_commodity(c, cat)
for m,s,d in MARKETS: create_market(m,s,d)

comms_df = read_commodities()
mkts_df  = read_markets()
days_back = 120
today     = date.today()
dates     = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_back)]

print(f"Generating {len(comms_df)} commodities × {len(mkts_df)} markets × {days_back} days...")
records = []
rng = np.random.default_rng(42)
for _, c in comms_df.iterrows():
    base = BASE_PRICES.get(c["name"], 2000)
    for _, m in mkts_df.iterrows():
        doys     = np.array([(today-timedelta(days=i)).timetuple().tm_yday for i in range(days_back)])
        seasonal = 1.0 + 0.15*np.sin(2*np.pi*(doys-90)/365)
        noise    = rng.normal(0, 0.03, days_back)
        premium  = {"Azadpur":1.05,"Vashi":1.08,"Bowenpally":0.97,"Koyambedu":1.02,"Yeshwanthpur":1.06}.get(m["market_name"],1.0)
        modal    = np.round(base*seasonal*premium*(1+noise), 2)
        spread   = rng.uniform(0.03, 0.07, days_back)
        min_p    = np.round(modal*(1-spread), 2)
        max_p    = np.round(modal*(1+spread), 2)
        arrivals = np.round(rng.uniform(50, 500, days_back), 1)
        for i, d in enumerate(dates):
            records.append((int(c["id"]),int(m["id"]),d,float(min_p[i]),float(max_p[i]),float(modal[i]),float(arrivals[i]),"Synthetic"))

print(f"Inserting {len(records):,} records into database...")
conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA journal_mode=WAL")
conn.executemany("""
    INSERT OR IGNORE INTO price_records
        (commodity_id,market_id,price_date,min_price,max_price,modal_price,arrivals,source)
    VALUES (?,?,?,?,?,?,?,?)
""", records)
conn.commit()
count = conn.execute("SELECT COUNT(*) FROM price_records").fetchone()[0]
conn.close()
print(f"Done! Database has {count:,} records.")
print("\nNow run:  streamlit run app.py")
