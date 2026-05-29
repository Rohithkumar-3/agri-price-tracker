"""
Test your Agmarknet API connection before running the full platform.
Usage: python test_api_connection.py
"""
import os
import sys
import requests
import json

def test_agmarknet(api_key: str):
    print("\n" + "="*60)
    print("  Agmarknet API Connection Test")
    print("="*60)

    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    # Test 1: Basic connectivity (no filters)
    print("\n[1] Basic connectivity test...")
    params = {"api-key": api_key, "format": "json", "limit": 5}
    try:
        resp = requests.get(url, params=params, timeout=10)
        print(f"    Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            total = data.get("total", "?")
            records = data.get("records", [])
            print(f"    ✓ Connected! Total records in dataset: {total:,}")
            if records:
                print(f"    Sample record keys: {list(records[0].keys())}")
                print(f"    Sample: {json.dumps(records[0], indent=6)}")
        elif resp.status_code == 401:
            print("    ✗ INVALID API KEY — check your key at data.gov.in")
            return False
        else:
            print(f"    ✗ Unexpected status: {resp.text[:200]}")
            return False
    except requests.exceptions.ConnectionError:
        print("    ✗ Cannot reach api.data.gov.in — check your internet connection")
        return False
    except requests.exceptions.Timeout:
        print("    ✗ Request timed out")
        return False

    # Test 2: Filter by state
    print("\n[2] Filter by state (Tamil Nadu)...")
    params["filters[state]"] = "Tamil Nadu"
    params["limit"] = 10
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code == 200:
        records = resp.json().get("records", [])
        print(f"    ✓ Found {len(records)} records for Tamil Nadu")
        if records:
            for r in records[:3]:
                comm = r.get("commodity", r.get("Commodity", "?"))
                mkt  = r.get("market",    r.get("Market",    "?"))
                mop  = r.get("modal_price", r.get("Modal Price", "?"))
                dat  = r.get("arrival_date", r.get("Arrival_Date", "?"))
                print(f"      {comm} @ {mkt}: ₹{mop} on {dat}")

    # Test 3: Filter by commodity
    print("\n[3] Filter by commodity (Tomato)...")
    params.pop("filters[state]", None)
    params["filters[commodity]"] = "Tomato"
    params["limit"] = 5
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code == 200:
        records = resp.json().get("records", [])
        print(f"    ✓ Found {len(records)} Tomato records")

    # Test 4: Detect field name format
    print("\n[4] Detecting API field names...")
    params = {"api-key": api_key, "format": "json", "limit": 1}
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code == 200:
        records = resp.json().get("records", [])
        if records:
            keys = list(records[0].keys())
            price_fields = [k for k in keys if "price" in k.lower() or "modal" in k.lower()]
            date_fields  = [k for k in keys if "date" in k.lower() or "arrival" in k.lower()]
            print(f"    Price fields found: {price_fields}")
            print(f"    Date fields found:  {date_fields}")
            print(f"    All fields: {keys}")

    print("\n✅ API is working correctly!")
    print("\nNext step: set your key in .env:")
    print(f"  echo 'AGMARKNET_API_KEY={api_key}' > .env")
    print("\nThen run the platform:")
    print("  streamlit run app.py")
    return True


def get_key_from_args_or_env():
    if len(sys.argv) > 1:
        return sys.argv[1]
    key = os.environ.get("AGMARKNET_API_KEY", "")
    if key:
        print(f"Using key from AGMARKNET_API_KEY env var: {key[:8]}...")
        return key
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("AGMARKNET_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"')
    return None


if __name__ == "__main__":
    key = get_key_from_args_or_env()
    if not key or key == "your_key_here":
        print("Usage:")
        print("  python test_api_connection.py YOUR_API_KEY")
        print("  OR set AGMARKNET_API_KEY=your_key in .env")
        print("\nGet your key at: https://data.gov.in → Login → My Account → Generate API Key")
        sys.exit(1)
    success = test_agmarknet(key)
    sys.exit(0 if success else 1)
