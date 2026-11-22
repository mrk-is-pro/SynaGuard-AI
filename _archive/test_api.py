"""Quick API checks. Run:  python test_api.py   (or: pytest test_api.py)"""
import pandas as pd
from fastapi.testclient import TestClient
import config
from api import app

client = TestClient(app)
row = pd.read_csv(config.CLEAN_DATA_FILE, parse_dates=["timestamp"]).query("is_anomaly == 0").iloc[500]
BASE = {k: (v.isoformat() if k == "timestamp" else v) for k, v in row[
    ["timestamp", "sku", "channel", "master_price", "channel_price", "master_stock", "channel_stock",
     "units_sold", "sync_delay_min"]].items()}
BASE = {k: (float(v) if k not in ("timestamp", "sku", "channel") else v) for k, v in BASE.items()}


def test_health():
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_loaded"]


def test_normal_record_not_flagged():
    r = client.post("/score", json=BASE).json()
    assert r["is_anomaly"] is False, r


def test_decimal_error_flagged_high():
    r = client.post("/score", json={**BASE, "channel_price": round(BASE["channel_price"] / 10, 2)}).json()
    assert r["is_anomaly"] and r["severity"] == "High" and r["likely_type"].startswith("Decimal"), r


def test_stale_sync_flagged():
    r = client.post("/score", json={**BASE, "sync_delay_min": 300}).json()
    assert r["is_anomaly"], r


def test_validation_rejects_negative_stock():
    assert client.post("/score", json={**BASE, "channel_stock": -5}).status_code == 422


def test_batch():
    recs = [BASE, {**BASE, "channel_price": 1.0}, {**BASE, "sync_delay_min": 400}]
    r = client.post("/score/batch", json={"records": recs, "only_flagged": True}).json()
    assert r["count"] == 3 and r["flagged"] >= 2 and len(r["results"]) == r["flagged"], r


def test_info_endpoints():
    for p in ("/model/info", "/thresholds", "/metrics"):
        assert client.get(p).status_code == 200


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
